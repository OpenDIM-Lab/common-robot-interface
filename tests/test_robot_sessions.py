"""Hardware-free construction, capture-session and approach policy contracts."""

import numpy as np
import pytest

from cri.robot import SyncRobot
from cri.topstar import TopstarSDKError
from cri.transforms import euler2quat
from test_topstar_controller import FakeX5


class Controller:
    def __init__(self, **options):
        self.options = options
        self.events = []

    def move_linear(self, pose, elbow=None):
        self.events.append(("linear", pose, elbow))

    def close(self):
        self.events.append("close")


def test_factory_uses_registry_and_preserves_options(monkeypatch):
    from cri import controller

    requested = []
    monkeypatch.setattr(controller, "get_controller", lambda name: requested.append(name) or Controller)
    options = {"ip": "example", "wrist": 1}
    robot = SyncRobot.from_config("example", options, axes="rxyz", tcp=[1, 2, 3, 4, 5, 6])
    assert requested == ["example"]
    assert options == robot.controller.options == {"ip": "example", "wrist": 1}
    assert robot.axes == "rxyz"
    np.testing.assert_allclose(robot.tcp, [1, 2, 3, 4, 5, 6])
    robot.close()


@pytest.mark.parametrize("axes,tcp", [("invalid", [0] * 6), ("sxyz", [0] * 5),
                                     ("sxyz", [0, 0, np.nan, 0, 0, 0])])
def test_factory_closes_controller_after_configuration_failure(monkeypatch, axes, tcp):
    from cri import controller

    backend = Controller()
    monkeypatch.setattr(controller, "get_controller", lambda name: lambda **kw: backend)
    with pytest.raises(ValueError):
        SyncRobot.from_config("example", {}, axes=axes, tcp=tcp)
    assert backend.events == ["close"]


@pytest.mark.parametrize("body_error", [None, RuntimeError("capture failed"), KeyboardInterrupt()])
def test_motion_session_finishes_then_closes_without_implicit_prepare(body_error):
    backend = Controller()
    backend.finish_motion = lambda: backend.events.append("finish")
    backend.prepare_motion = lambda: pytest.fail("session must not enable implicitly")
    robot = SyncRobot(backend)

    def run():
        with robot.motion_session() as active:
            assert active is robot
            if body_error is not None:
                raise body_error

    if body_error is None:
        run()
    else:
        with pytest.raises(type(body_error)) as caught:
            run()
        assert caught.value is body_error
    assert backend.events == ["finish", "close"]


def test_failed_finish_still_closes_and_keeps_body_exception_context():
    backend = Controller()
    primary = RuntimeError("motion failed")
    cleanup = ValueError("finish failed")

    def finish():
        backend.events.append("finish")
        raise cleanup

    backend.finish_motion = finish
    with pytest.raises(ValueError) as caught:
        with SyncRobot(backend).motion_session():
            raise primary
    assert caught.value is cleanup and cleanup.__context__ is primary
    assert backend.events == ["finish", "close"]


def test_optional_finish_and_regular_read_only_context():
    backend = Controller()
    with SyncRobot(backend).motion_session():
        pass
    assert backend.events == ["close"]
    backend.finish_motion = lambda: pytest.fail("read-only context must not finish motion")
    with SyncRobot(backend):
        pass
    assert backend.events == ["close", "close"]


def test_approach_fallback_is_the_same_single_linear_move():
    backend = Controller()
    robot = SyncRobot(backend)
    robot.tcp = [10, -20, 30, 21, 32, -43]
    robot.coord_frame = [50, 60, 70, -25, 16, 7]
    target = [80, 90, 100, 11, 12, 13]
    robot.move_linear(target)
    robot.move_linear_approach(target, previous_pose=[0] * 6)
    assert len(backend.events) == 2
    np.testing.assert_allclose(backend.events[0][1], backend.events[1][1])
    assert backend.events[0][2] is backend.events[1][2] is None


def test_approach_backend_failure_is_not_retried_as_ordinary_motion():
    backend = Controller()

    def approach(pose, previous):
        assert previous is None
        raise NotImplementedError("specific approach failed")

    backend.move_linear_approach = approach
    with pytest.raises(NotImplementedError, match="specific approach failed"):
        SyncRobot(backend).move_linear_approach([0] * 6)
    assert backend.events == []


def topstar(sdk, **options):
    return SyncRobot.from_config(
        "topstar", dict(sdk=sdk, euler_axes="sxyz", state_timeout=.1, **options),
        axes="sxyz", tcp=[0] * 6,
    )


@pytest.mark.parametrize("wrist", [None, 0, 1])
@pytest.mark.parametrize("previous_rotation", [None, [10, 20, 30], [10, 20, 390],
                                               [10, 20, 30.00001], [10, 20, 30.001], [-20, 15, 40]])
def test_topstar_approach_matches_original_wrist_policy_in_rotated_frames(wrist, previous_rotation):
    sdk = FakeX5()
    with topstar(sdk, **({} if wrist is None else {"wrist": wrist})) as robot:
        robot.tcp = [14, -3, 65, -11, 28, 34]
        robot.coord_frame = [20, 30, 40, 15, -23, 52]
        target = [310, 35, 455, 10, 20, 30]
        previous = None if previous_rotation is None else [300, 25, 450, *previous_rotation]
        expected = 0
        if wrist != 0 and previous is not None:
            if abs(np.dot(euler2quat(previous)[3:], euler2quat(target)[3:])) < 1 - 1e-12:
                expected = 1
        robot.move_linear(target)
        original = sdk.named("movl")[-1][1]
        robot.move_linear_approach(target, previous)
        actual = sdk.named("movl")[-1][1]
        assert vars(actual[2]) == {**vars(original[2]), "wrist": expected}
        np.testing.assert_allclose(actual[1].pose.tolist(), original[1].pose.tolist())
        assert (actual[1].uf, actual[1].tf, actual[1].cfg) == (original[1].uf, original[1].tf, original[1].cfg)
        robot.move_linear(target)
        assert sdk.named("movl")[-1][1][2].wrist == 0
        robot.move_linear(target, wrist=1)
        assert sdk.named("movl")[-1][1][2].wrist == 1


def test_topstar_approach_accepts_equivalent_quaternion_sign():
    sdk = FakeX5()
    with topstar(sdk) as robot:
        pose = euler2quat([310, 35, 455, 10, 20, 30])
        previous = pose.copy()
        previous[3:] *= -1
        robot.controller.move_linear_approach(pose, previous)
        assert sdk.named("movl")[-1][1][2].wrist == 0


@pytest.mark.parametrize("wrist", [-1, 2, True, 0.5])
def test_invalid_approach_wrist_is_rejected_before_connection(wrist):
    sdk = FakeX5()
    with pytest.raises(ValueError, match="wrist"):
        topstar(sdk, wrist=wrist)
    assert not sdk.calls


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_topstar_session_attempts_servo_off_before_disconnect_even_on_failure(fail_cleanup):
    sdk = FakeX5()
    robot = topstar(sdk)
    if fail_cleanup:
        sdk.fail["enable_servo"] = RuntimeError("off failed")
        with pytest.raises(TopstarSDKError):
            with robot.motion_session():
                pass
    else:
        with robot.motion_session():
            pass
        assert sdk.state.enable == 0
    operations = [call[0] for call in sdk.calls]
    assert operations.index("enable_servo") < operations.index("disconnect")
    assert sdk.named("enable_servo")[-1][1][1] is False
    assert not sdk.named("set_system_mode")


def test_topstar_factory_and_read_only_preflight_never_enable_or_disable():
    sdk = FakeX5()
    sdk.state.mode, sdk.state.enable = 3, 0
    with topstar(sdk) as robot:
        with robot.linear_path_checker() as check:
            check(([310, 35, 455, 10, 20, 30],))
    assert sdk.named("cnvrt_j") and sdk.named("disconnect")
    for name in ("enable_servo", "set_system_mode", "movl", "movj", "movc"):
        assert not sdk.named(name)
