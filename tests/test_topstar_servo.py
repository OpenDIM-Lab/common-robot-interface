"""Native servol/servoj tests, including cancellation and watchdog cleanup."""

import copy
import threading

import numpy as np
import pytest

from cri.robot import AsyncRobot, SyncRobot
from cri.topstar import TopstarController, TopstarSDKError, TopstarStateError, TopstarTimeoutError
from cri.transforms import euler2quat
from test_topstar_controller import FakeX5


class ServoX5(FakeX5):
    def servol(self, handle, point, cmdt, lookahead, gain, vel, acc):
        self._call("servol", handle, point, cmdt, lookahead, gain, vel, acc)
        self.point = copy.deepcopy(point)
        self.state.state, self.state.in_pos = 1, 0

    def servoj(self, handle, joint, cmdt, lookahead, gain, vel, acc):
        self._call("servoj", handle, joint, cmdt, lookahead, gain, vel, acc)
        self.joints = copy.deepcopy(joint)
        self.state.state, self.state.in_pos = 1, 0


@pytest.fixture
def pair():
    sdk = ServoX5()
    controller = TopstarController(sdk=sdk, euler_axes="sxyz", command_speed=2, state_timeout=0.1)
    try:
        yield controller, sdk
    finally:
        sdk.fail.clear()
        controller.close()


def test_servo_is_explicit_and_uses_native_sdk_without_arrival_wait(pair):
    controller, sdk = pair
    pose = euler2quat([310, 25, 450, 10, 20, 30])
    with pytest.raises(TopstarStateError, match="prepare_servo"):
        controller.servo_move_linear(pose)
    controller.prepare_motion()
    controller.prepare_servo(period=1 / 30, gain=5, acceleration=12)
    controller.servo_move_linear(pose)
    controller.servo_move_linear(pose)  # Robot is moving; updates must not require idle.
    args = sdk.named("servol")[-1][1]
    assert (args[1].uf, args[1].tf, args[1].cfg) == (0, 0, sdk.point.cfg)
    assert args[2:] == (1 / 30, 0, 5, 2, 12)
    assert not sdk.named("movl") and not sdk.named("wait_move_done")
    assert controller.last_motion_completed is False
    np.testing.assert_allclose(controller.commanded_pose, pose)
    for action in (lambda: controller.move_linear(pose), controller.prepare_motion,
                   lambda: setattr(controller, "tcp", euler2quat([0] * 6))):
        with pytest.raises(TopstarStateError):
            action()


def test_joint_servo_preserves_external_axes_and_checks_limits(pair):
    controller, sdk = pair
    sdk.joints.e1 = 17
    controller.prepare_servo()
    controller.servo_move_joints([1, 2, 3, 4, 5, 6])
    args = sdk.named("servoj")[-1][1]
    assert args[1].tolist() == [1, 2, 3, 4, 5, 6, 17, 0, 0]
    np.testing.assert_allclose(controller.commanded_joint_angles, [1, 2, 3, 4, 5, 6])
    assert not sdk.named("movj")


@pytest.mark.parametrize("options", [
    {"period": 0.001}, {"period": 3}, {"period": float("nan")},
    {"gain": 51}, {"gain": 2.5}, {"acceleration": 0}, {"acceleration": 101},
    {"watchdog_timeout": 0.01}, {"watchdog_timeout": float("inf")},
])
def test_invalid_servo_settings_do_not_start_session(pair, options):
    controller, sdk = pair
    with pytest.raises(ValueError):
        controller.prepare_servo(**options)
    assert not controller.servo_active and not sdk.named("servol")


def test_missing_optional_sdk_servo_is_not_silently_replaced_with_movl(pair):
    controller, sdk = pair
    sdk.servol = None
    with pytest.raises(NotImplementedError, match="servol"):
        controller.prepare_servo()
    assert not controller.servo_active and not sdk.named("movl")


def test_servo_does_not_silently_ignore_a_nonzero_blend_setting(pair):
    controller, sdk = pair
    controller._blend_mapping[5] = 20
    controller.blend_radius = 5
    with pytest.raises(ValueError, match="blend_radius"):
        controller.prepare_servo()
    assert not controller.servo_active
    controller.blend_radius = 0
    controller.prepare_servo()
    with pytest.raises(ValueError, match="blend_radius"):
        controller.blend_radius = 5


@pytest.mark.parametrize("field,value", [
    ("alarm", 1), ("enable", 0), ("remote", 1), ("rgm_state", 1),
    ("mode", 2), ("state", 2),
])
def test_bad_state_before_stream_update_aborts_instead_of_submitting(pair, field, value):
    controller, sdk = pair
    controller.prepare_servo()
    setattr(sdk.state, field, value)
    with pytest.raises(TopstarStateError):
        controller.servo_move_joints([0] * 6)
    assert not sdk.named("servoj")
    assert sdk.named("abort")
    assert not controller.servo_active


def test_watchdog_aborts_latches_and_failure_is_visible_on_next_call(pair):
    controller, sdk = pair
    stopped = threading.Event()
    abort = sdk.abort

    def record_stop(handle):
        abort(handle)
        stopped.set()

    sdk.abort = record_stop
    controller.prepare_servo(period=0.02, watchdog_timeout=0.06)
    controller.servo_move_linear(euler2quat([300, 25, 450, 10, 20, 30]))
    assert stopped.wait(1)
    assert isinstance(controller.servo_error, TopstarTimeoutError)
    with pytest.raises(TopstarTimeoutError):
        controller.servo_move_joints([0] * 6)
    assert not controller.servo_active
    sdk.state.state = 3
    controller.prepare_motion()
    assert controller.servo_error is None
    controller.prepare_servo()


def test_stop_cancels_servo_and_close_stops_pending_target(pair):
    controller, sdk = pair
    controller.prepare_servo()
    controller.servo_move_joints([0] * 6)
    controller.stop()
    assert sdk.named("stop") and not controller.servo_active
    with pytest.raises(TopstarStateError):
        controller.prepare_servo()
    sdk.state.state = 3
    controller.prepare_motion()
    controller.prepare_servo()
    controller.servo_move_joints([0] * 6)
    controller.close()
    assert sdk.named("abort") and not controller.connected


def test_stop_during_ik_prevents_servo_submission(pair):
    controller, sdk = pair
    controller.prepare_servo()
    original = sdk.cnvrt_j

    def inverse(*args):
        result = original(*args)
        controller.stop()
        return result

    sdk.cnvrt_j = inverse
    with pytest.raises(Exception, match="cancelled"):
        controller.servo_move_linear(euler2quat([300, 25, 450, 10, 20, 30]))
    assert not sdk.named("servol")


def test_late_target_cannot_refresh_expired_deadline_before_watchdog_runs(pair):
    controller, sdk = pair
    controller.prepare_servo()
    controller.servo_move_joints([0] * 6)
    with controller._sdk_lock:
        controller._servo_deadline = -1
        with pytest.raises(TopstarTimeoutError):
            controller.servo_move_joints([1] * 6)
    assert len(sdk.named("servoj")) == 1
    assert sdk.named("abort")


def test_servo_disable_stops_pending_target_before_switching_off(pair):
    controller, sdk = pair
    controller.prepare_servo()
    controller.servo_move_joints([0] * 6)
    controller.disable_servo()
    assert sdk.named("abort")
    assert sdk.state.enable == 0
    assert not controller.servo_active


def test_async_vendor_error_is_reported_and_stops_stream(pair):
    controller, sdk = pair
    controller.prepare_servo()
    sdk.fail["servol"] = RuntimeError("native servo failed")
    robot = AsyncRobot(SyncRobot(controller))
    robot.async_servo_move_linear([300, 25, 450, 10, 20, 30])
    with pytest.raises(TopstarSDKError, match="native servo failed"):
        robot.async_result(timeout=1)
    assert sdk.named("abort") and not controller.servo_active
    robot.close()


def test_hardware_tool_cannot_be_combined_with_nonzero_software_tcp():
    sdk = ServoX5()
    with TopstarController(sdk=sdk, tool_index=1, euler_axes="sxyz") as controller:
        robot = SyncRobot(controller)
        with pytest.raises(ValueError, match="identity"):
            robot.tcp = [0, 0, 50, 0, 0, 0]
        assert not sdk.named("set_tf")


def test_native_servo_uses_only_command_speed_through_async_wrapper():
    sdk = ServoX5()
    controller = TopstarController(sdk=sdk, euler_axes="sxyz", command_speed=4)
    with AsyncRobot(SyncRobot(controller)) as robot:
        robot.command_speed = 6
        robot.prepare_servo()
        robot.async_servo_move_linear([300, 25, 450, 10, 20, 30])
        robot.async_result(timeout=1)
        robot.servo_move_joints([0] * 6)
        assert sdk.named("servol")[-1][1][-2] == 6
        assert sdk.named("servoj")[-1][1][-2] == 6
        assert not sdk.named("set_speed")
