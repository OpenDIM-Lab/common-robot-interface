"""Offline regression tests for software tools and async worker failures."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cri.robot import AsyncBusy, AsyncNotBusy, AsyncRobot, SyncRobot
from cri.transforms import euler2mat, euler2quat, quat2mat


class RecordingController:
    def __init__(self):
        self.pose = euler2quat([0] * 6)
        self.commanded_pose = self.pose.copy()
        self.circular_moves = []

    def move_linear(self, pose, elbow=None):
        self.pose = self.commanded_pose = np.array(pose)

    def move_circular(self, via, end, elbow=None):
        self.circular_moves.append((via, end))
        self.move_linear(end)

    def close(self):
        pass


def test_positive_tool_offset_read_and_target_have_conventional_direction():
    controller = RecordingController()
    robot = SyncRobot(controller)
    robot.tcp = [0, 0, 50, 0, 0, 0]
    assert robot.pose[2] == 50
    robot.move_linear([0, 0, 100, 0, 0, 0])
    assert controller.pose[2] == 50
    assert robot.pose[2] == robot.target_pose[2] == 100
    assert not hasattr(robot, "controller_tcp")


@pytest.mark.parametrize("axes", ["sxyz", "rxyz"])
def test_mixed_rotation_tool_and_work_frame_match_independent_matrix_composition(axes):
    controller = RecordingController()
    robot = SyncRobot(controller)
    robot.axes = axes
    work = [100, -20, 50, 23, -31, 47]
    tool = [15, -10, 70, -14, 27, 33]
    target = [12, 24, -5, 36, 19, -42]
    robot.coord_frame, robot.tcp = work, tool
    expected = euler2mat(work, axes) @ euler2mat(target, axes) @ np.linalg.inv(euler2mat(tool, axes))
    robot.move_linear(target)
    np.testing.assert_allclose(quat2mat(controller.pose), expected, atol=1e-10)
    np.testing.assert_allclose(euler2mat(robot.pose, axes), euler2mat(target, axes), atol=1e-10)
    np.testing.assert_allclose(euler2mat(robot.target_pose, axes), euler2mat(target, axes), atol=1e-10)
    via = [2, 3, 4, -12, 21, 36]
    robot.move_circular(via, target)
    actual_via, actual_end = controller.circular_moves[-1]
    np.testing.assert_allclose(quat2mat(actual_via),
                               euler2mat(work, axes) @ euler2mat(via, axes) @ np.linalg.inv(euler2mat(tool, axes)),
                               atol=1e-10)
    np.testing.assert_allclose(quat2mat(actual_end), expected, atol=1e-10)


def test_software_tool_never_writes_persistent_hardware_tool():
    class Controller(RecordingController):
        @property
        def tcp(self):
            return euler2quat([0] * 6)

        @tcp.setter
        def tcp(self, value):
            raise AssertionError("software TCP must not write a hardware tool")

    robot = SyncRobot(Controller())
    robot.tcp = [10, 20, 30, 11, 22, 33]


@pytest.mark.parametrize("method,args", [
    ("async_move_linear", ([0] * 6,)),
    ("async_move_joints", ([0] * 6,)),
    ("async_move_circular", ([0] * 6, [1] * 6)),
    ("async_servo_move_linear", ([0] * 6,)),
    ("async_servo_move_joints", ([0] * 6,)),
])
def test_async_failure_is_rethrown_and_busy_state_can_be_reused(method, args):
    failure = RuntimeError("vendor rejected target")

    def fail(*args):
        raise failure

    name = method.removeprefix("async_")
    sync = SimpleNamespace(close=lambda: None, **{name: fail})
    with AsyncRobot(sync) as robot:
        for _ in range(2):
            getattr(robot, method)(*args)
            with pytest.raises(RuntimeError) as caught:
                robot.async_result(timeout=1)
            assert caught.value is failure
            with pytest.raises(AsyncNotBusy):
                robot.async_done()


def test_async_timeout_keeps_pending_result_and_close_joins_worker():
    started, release = threading.Event(), threading.Event()

    def move(pose, elbow):
        started.set()
        release.wait(2)
        return 42

    sync = SimpleNamespace(move_linear=move, close=release.set)
    robot = AsyncRobot(sync)
    robot.async_move_linear([0] * 6)
    assert started.wait(1)
    with pytest.raises(TimeoutError):
        robot.async_result(timeout=0.001)
    assert not robot.async_done()
    with pytest.raises(AsyncBusy):
        robot.async_move_linear([0] * 6)
    robot.close()
    assert robot.async_result(timeout=1) == 42
    with pytest.raises(RuntimeError, match="closed"):
        robot.async_move_linear([0] * 6)


def test_lifecycle_and_native_servo_are_forwarded_through_both_wrappers():
    calls = []
    controller = RecordingController()
    for name in ("prepare_motion", "prepare_servo", "stop", "abort", "disable_servo",
                 "servo_move_linear", "servo_move_joints"):
        setattr(controller, name, lambda *args, _name=name, **kw: calls.append((_name, args, kw)))
    with AsyncRobot(SyncRobot(controller)) as robot:
        robot.tcp = [0, 0, 10, 0, 0, 0]
        robot.prepare_motion()
        robot.prepare_servo(period=0.032)
        robot.servo_move_linear([0, 0, 30, 0, 0, 0])
        assert calls[-1][1][0][2] == 20
        robot.servo_move_joints([1] * 6)
        robot.async_servo_move_linear([0, 0, 40, 0, 0, 0])
        robot.async_result(timeout=1)
        assert calls[-1][1][0][2] == 30
        robot.stop()
        robot.abort()
        robot.disable_servo()
    assert [c[0] for c in calls] == ["prepare_motion", "prepare_servo", "servo_move_linear",
                                    "servo_move_joints", "servo_move_linear", "stop", "abort", "disable_servo"]
