"""Hardware-free contract tests for the Topstar X5 backend.

The fake models the supplied X5 Python 3.2.5 SDK's public structures and
exception/return semantics.  It is test support, never a vendor SDK replacement.
"""

import copy
import math
import subprocess
import sys
import threading
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cri.controller import RobotController
from cri.robot import SyncRobot
from cri.transforms import euler2quat, quat2mat
from cri.topstar.topstar_controller import (
    TopstarController,
    TopstarError,
    TopstarSDKError,
    TopstarStateError,
    TopstarMotionError,
    TopstarTimeoutError,
)


class FakePose:
    def __init__(self, x=0., y=0., z=0., a=0., b=0., c=0.,
                 e1=0., e2=0., e3=0., **kwargs):
        for name, value in zip(('x', 'y', 'z', 'a', 'b', 'c', 'e1', 'e2', 'e3'),
                               (x, y, z, a, b, c, e1, e2, e3)):
            setattr(self, name, float(value))

    def tolist(self):
        return [getattr(self, key) for key in ('x', 'y', 'z', 'a', 'b', 'c', 'e1', 'e2', 'e3')]


class FakeJoint:
    def __init__(self, j1=0., j2=0., j3=0., j4=0., j5=0., j6=0.,
                 e1=0., e2=0., e3=0., **kwargs):
        for name, value in zip(('j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'e1', 'e2', 'e3'),
                               (j1, j2, j3, j4, j5, j6, e1, e2, e3)):
            setattr(self, name, float(value))

    def tolist(self):
        return [getattr(self, key) for key in ('j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'e1', 'e2', 'e3')]


class FakePoint:
    def __init__(self, pose=None, uf=0, tf=0, cfg=(0, 0, 0, 1), **kwargs):
        self.pose = pose if isinstance(pose, FakePose) else FakePose(*(pose or ()))
        self.uf = int(uf)
        self.tf = int(tf)
        self.cfg = tuple(cfg)


class FakeMovPointAdd:
    def __init__(self, vel=100, cnt=0, acc=100, dec=100, offset=-2,
                 toffset=-2, vision_num=-1, track_num=-1, wrist=0, inc=0,
                 search=0, rtcp=0, **kwargs):
        self.__dict__.update(locals())
        self.__dict__.pop('self')


class FakeX5:
    Pose = FakePose
    Joint = FakeJoint
    Point = FakePoint
    MovPointAdd = FakeMovPointAdd
    SYSTEM_MODE_MANUAL = 3
    SYSTEM_MODE_AUTOCMD = 100

    def __init__(self):
        self.calls = []
        self.fail = {}
        self.handle = 7
        self.joints = FakeJoint(1, -2, 3, -4, 5, -6)
        self.point = FakePoint((300, 25, 450, 10, 20, 30), 0, 0, (1, 0, 1, 1))
        self.tools = {i: FakePose() for i in range(17)}
        self.state = SimpleNamespace(
            link=1, enable=1, alarm=0, mode=100, state=3, in_pos=1,
            remote=0, cmd_queue=0, rgm_state=0, speed=10, uf_no=0, tf_no=0,
            user_point=self.point, world_point=self.point, joint=self.joints,
            di_list=[0] * 32, do_list=[0] * 32, ei_list=[0] * 8,
        )
        self.move_states = []
        self.system_states = []
        self.alarm_info = []
        self.default_move_state = 1
        self.converted_point = None
        self.ik_joints = FakeJoint(11, -22, 33, -44, 55, -66)
        self.post_submission_state = {}
        self.submitted = threading.Event()
        self.mode_callback = None
        self.move_state_callback = None
        self.wait_callback = None

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, copy.deepcopy(args), copy.deepcopy(kwargs)))
        error = self.fail.get(name)
        if error:
            raise error

    def named(self, name):
        return [call for call in self.calls if call[0] == name]

    def connect(self, ip, link_timeout=2, retry_num=5):
        self._call('connect', ip, link_timeout, retry_num)
        return self.handle

    def disconnect(self, handle):
        self._call('disconnect', handle)

    def get_version(self, handle):
        self._call('get_version', handle)
        return 'XMotion 5.6.9 (fake)'

    def get_all_system_state(self, handle):
        self._call('get_all_system_state', handle)
        value = self.system_states.pop(0) if self.system_states else self.state
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)

    def get_system_alarm_info(self, handle):
        self._call('get_system_alarm_info', handle)
        return copy.deepcopy(self.alarm_info)

    def get_cjoint(self, handle):
        self._call('get_cjoint', handle)
        return copy.deepcopy(self.joints)

    def get_wpoint(self, handle):
        self._call('get_wpoint', handle)
        return copy.deepcopy(self.point)

    def get_cpoint(self, handle):
        self._call('get_cpoint', handle)
        return copy.deepcopy(self.point)

    def get_tf(self, handle, index):
        self._call('get_tf', handle, index)
        return copy.deepcopy(self.tools[index])

    def set_tf(self, handle, index, data):
        self._call('set_tf', handle, index, data)
        self.tools[index] = copy.deepcopy(data)

    def set_speed(self, handle, scale):
        self._call('set_speed', handle, scale)
        self.state.speed = scale

    def get_speed(self, handle):
        self._call('get_speed', handle)
        return self.state.speed

    def set_system_mode(self, handle, mode):
        self._call('set_system_mode', handle, mode)
        self.state.mode = mode
        if self.mode_callback:
            self.mode_callback()

    def enable_servo(self, handle, is_servo_on):
        self._call('enable_servo', handle, is_servo_on)
        self.state.enable = int(is_servo_on)

    def cnvrt_p(self, handle, src_joint, uf, tf):
        self._call('cnvrt_p', handle, src_joint, uf, tf)
        # Deliberately deterministic rather than pretending to model kinematics.
        return FakePoint((101, 202, 303, 10, 20, 30), uf, tf, self.point.cfg)

    def cnvrt_j(self, handle, src_point, inverse_type, last_joint):
        self._call('cnvrt_j', handle, src_point, inverse_type, last_joint)
        return copy.deepcopy(self.ik_joints)

    def cnvrt_p_to_p(self, handle, source_point, target_uf, target_tf):
        self._call('cnvrt_p_to_p', handle, source_point, target_uf, target_tf)
        if self.converted_point is not None:
            return copy.deepcopy(self.converted_point)
        result = copy.deepcopy(source_point)
        result.uf = target_uf
        result.tf = target_tf
        return result

    def movj(self, handle, point, additional_data=None):
        self._call('movj', handle, point, additional_data)
        self.joints = copy.deepcopy(point)
        self.state.joint = self.joints
        self.state.__dict__.update(self.post_submission_state)
        self.submitted.set()

    def movl(self, handle, point, additional_data=None):
        self._call('movl', handle, point, additional_data)
        self.point = copy.deepcopy(point)
        self.state.world_point = self.point
        self.state.__dict__.update(self.post_submission_state)

    def movc(self, handle, middle_point, target_point, central_angle=None,
             additional_data=None):
        self._call('movc', handle, middle_point, target_point,
                   central_angle, additional_data)
        self.point = copy.deepcopy(target_point)
        self.state.world_point = self.point

    def get_cur_move_state(self, handle):
        self._call('get_cur_move_state', handle)
        if self.move_state_callback:
            self.move_state_callback()
        return self.move_states.pop(0) if self.move_states else self.default_move_state

    def wait_move_done(self, handle, timeout_ms=-1):
        self._call('wait_move_done', handle, timeout_ms)
        if self.wait_callback:
            self.wait_callback()

    def wait_cmd_send_done(self, handle):
        self._call('wait_cmd_send_done', handle)

    def stop(self, handle):
        self._call('stop', handle)
        self.default_move_state = 2
        self.state.in_pos = 1
        self.state.cmd_queue = 0

    def abort(self, handle):
        self._call('abort', handle)
        self.default_move_state = 2
        self.state.in_pos = 1
        self.state.cmd_queue = 0


class TopstarControllerTests(unittest.TestCase):
    def setUp(self):
        self.sdk = FakeX5()
        self.controllers = []

    def tearDown(self):
        self.sdk.fail.pop('disconnect', None)
        for controller in self.controllers:
            try:
                controller.close()
            except TopstarError:
                pass

    def controller(self, **kwargs):
        defaults = dict(sdk=self.sdk, euler_axes='sxyz',
                        poll_interval=.02, move_timeout=.1, state_timeout=.1)
        defaults.update(kwargs)
        instance = TopstarController(ip='192.0.2.10', **defaults)
        self.controllers.append(instance)
        return instance

    def last_motion(self, name):
        return self.sdk.named(name)[-1][1]

    def test_controller_is_concrete_robot_controller(self):
        controller = self.controller()
        self.assertIsInstance(controller, RobotController)
        self.assertFalse(TopstarController.__abstractmethods__)

    def test_connect_is_read_only_and_info_identifies_endpoint_and_version(self):
        controller = self.controller()
        self.assertEqual(len(self.sdk.named('connect')), 1)
        self.assertIn('192.0.2.10', controller.info)
        self.assertIn('5.6.9', controller.info)
        for name in ('enable_servo', 'set_system_mode', 'set_speed', 'set_tf', 'movj', 'movl', 'movc'):
            self.assertFalse(self.sdk.named(name), name)

    def test_no_sdk_is_required_to_import_backend(self):
        result = subprocess.run(
            [sys.executable, '-c',
             "import sys; from cri.controller import get_controller; "
             "get_controller('topstar'); assert 'xapi.api' not in sys.modules"],
            check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_close_is_idempotent_and_context_manager_disconnects(self):
        with self.controller() as controller:
            self.assertIsNotNone(controller.info)
        controller.close()
        self.assertEqual(len(self.sdk.named('disconnect')), 1)

    def test_constructor_failure_disconnects_acquired_handle(self):
        self.sdk.fail['get_version'] = RuntimeError('version read failed')
        with self.assertRaises(TopstarSDKError):
            self.controller()
        self.assertEqual(len(self.sdk.named('disconnect')), 1)

    def test_zero_handle_is_valid_and_disconnected_once(self):
        self.sdk.handle = 0
        controller = self.controller()
        self.assertTrue(controller.connected)
        controller.close()
        self.assertEqual(self.sdk.named('disconnect')[-1][1], (0,))

    def test_disconnect_failure_still_releases_adapter_handle(self):
        controller = self.controller()
        self.sdk.fail['disconnect'] = RuntimeError('disconnect failed')
        with self.assertRaises(TopstarSDKError):
            controller.close()
        self.assertFalse(controller.connected)
        controller.close()
        self.assertEqual(len(self.sdk.named('disconnect')), 1)

    def test_closed_controller_rejects_feedback_and_motion(self):
        controller = self.controller()
        controller.close()
        with self.assertRaises(TopstarError):
            _ = controller.joint_angles
        with self.assertRaises(TopstarError):
            controller.move_joints([0] * 6)

    def test_joint_order_degrees_and_defensive_copies(self):
        controller = self.controller()
        assert_allclose(controller.joint_angles, (1, -2, 3, -4, 5, -6))
        value = controller.joint_angles
        value[0] = 999
        self.assertEqual(controller.joint_angles[0], 1)
        assert_allclose(controller.commanded_joint_angles, (1, -2, 3, -4, 5, -6))
        self.assertIsNone(controller.elbow)
        self.assertIsNone(controller.commanded_elbow)

    def test_pose_is_world_mm_and_wxyz_with_mixed_rotation(self):
        controller = self.controller()
        expected = euler2quat((300, 25, 450, 10, 20, 30), 'sxyz')
        assert_allclose(quat2mat(controller.pose), quat2mat(expected), atol=1e-10)
        assert_allclose(controller.commanded_pose, expected, atol=1e-10)
        self.assertFalse(self.sdk.named('get_cpoint'))

    def test_mismatched_active_frame_is_converted_explicitly(self):
        self.sdk.point = FakePoint((1, 2, 3, 0, 0, 0), 5, 4)
        self.sdk.state.world_point = self.sdk.point
        self.sdk.converted_point = FakePoint((310, 40, 500, 0, 0, 90), 0, 0)
        controller = self.controller()
        assert_allclose(controller.pose[:3], (310, 40, 500))
        self.assertTrue(self.sdk.named('cnvrt_p_to_p'))
        args = self.sdk.named('cnvrt_p_to_p')[-1][1]
        self.assertEqual(args[-2:], (0, 0))

    def test_unknown_euler_convention_allows_diagnostics_but_blocks_pose(self):
        controller = self.controller(euler_axes=None)
        self.assertIn('5.6.9', controller.info)
        self.assertEqual(len(controller.joint_angles), 6)
        with self.assertRaises((TopstarError, ValueError)):
            _ = controller.pose
        with self.assertRaises((TopstarError, ValueError)):
            controller.move_linear((300, 0, 400, 1, 0, 0, 0))
        self.assertFalse(self.sdk.named('movl'))

    def test_identity_tool_zero_is_never_written(self):
        controller = self.controller()
        assert_allclose(controller.tcp, (0, 0, 0, 1, 0, 0, 0))
        controller.tcp = (0, 0, 0, -1, 0, 0, 0)
        with self.assertRaises((TopstarError, ValueError)):
            controller.tcp = (0, 0, 10, 1, 0, 0, 0)
        self.assertFalse(self.sdk.named('set_tf'))

    def test_configured_tcp_reads_and_writes_selected_tool(self):
        self.sdk.tools[2] = FakePose(0, 0, 70, 30, 10, -20)
        controller = self.controller(tool_index=2)
        expected = euler2quat((0, 0, 70, 30, 10, -20))
        assert_allclose(quat2mat(controller.tcp), quat2mat(expected), atol=1e-10)
        target = euler2quat((5, 10, 80, -20, 25, 40))
        controller.tcp = target
        args = self.sdk.named('set_tf')[-1][1]
        self.assertEqual(args[1], 2)
        assert_allclose(args[2].tolist()[:6], (5, 10, 80, -20, 25, 40), atol=1e-10)

    def test_move_joints_preserves_joint_order(self):
        controller = self.controller(command_speed=12)
        target = np.array([2., -3, 4, -5, 6, -7])
        controller.move_joints(target)
        args = self.last_motion('movj')
        assert_allclose(args[1].tolist()[:6], target)
        self.assertEqual(args[2].vel, 12)
        self.assertEqual(args[2].cnt, -1)
        assert_allclose(controller.commanded_joint_angles, target)
        target[:] = 100
        self.assertNotEqual(controller.commanded_joint_angles[0], 100)
        self.assertTrue(self.sdk.named('cnvrt_p'))

    def test_joint_motion_preserves_current_external_axes(self):
        self.sdk.joints = FakeJoint(1, -2, 3, -4, 5, -6, 17, 28, 39)
        controller = self.controller()
        controller.move_joints([0] * 6)
        assert_allclose(self.last_motion('movj')[1].tolist()[6:], (17, 28, 39))

    def test_cartesian_motion_preserves_external_axes_and_configuration(self):
        self.sdk.point.pose.e1, self.sdk.point.pose.e2, self.sdk.point.pose.e3 = (7, 8, 9)
        controller = self.controller()
        controller.move_linear(euler2quat((310, 35, 455, -15, 25, 40)))
        target = self.last_motion('movl')[1]
        assert_allclose(target.pose.tolist()[6:], (7, 8, 9))
        self.assertEqual(target.cfg, (1, 0, 1, 1))

    def test_linear_target_conversion_ik_and_commanded_state(self):
        controller = self.controller(command_speed=12)
        target = euler2quat((310, 35, 455, -15, 25, 40))
        controller.move_linear(target)
        args = self.last_motion('movl')
        self.assertEqual((args[1].uf, args[1].tf), (0, 0))
        assert_allclose(args[1].pose.tolist()[:6], (310, 35, 455, -15, 25, 40), atol=1e-10)
        self.assertEqual(args[2].vel, 12)
        self.assertEqual(self.sdk.named('cnvrt_j')[-1][1][2], 1)
        assert_allclose(controller.commanded_pose, target, atol=1e-10)
        assert_allclose(controller.commanded_joint_angles, self.sdk.ik_joints.tolist()[:6])

    def test_linear_wrist_option_changes_only_the_sdk_wrist_field(self):
        robot = SyncRobot(self.controller(command_speed=12, command_acceleration=5,
                                          command_deceleration=8))
        target = (310, 35, 455, -15, 25, 40)
        robot.move_linear(target)
        _, original_point, original_data = self.last_motion('movl')
        robot.move_linear(target, wrist=1)
        _, wrist_point, wrist_data = self.last_motion('movl')

        self.assertEqual(wrist_point.pose.tolist(), original_point.pose.tolist())
        self.assertEqual((wrist_point.uf, wrist_point.tf, wrist_point.cfg),
                         (original_point.uf, original_point.tf, original_point.cfg))
        self.assertEqual(original_data.wrist, 0)
        self.assertEqual(vars(wrist_data), {**vars(original_data), 'wrist': 1})
        robot.move_linear(target)
        self.assertEqual(self.last_motion('movl')[2].wrist, 0)

    def test_linear_wrist_rejects_an_unsupported_value_before_submission(self):
        controller = self.controller()
        target = euler2quat((310, 35, 455, -15, 25, 40))
        with self.assertRaisesRegex(ValueError, 'wrist'):
            controller.move_linear(target, wrist=2)
        self.assertFalse(self.sdk.named('movl'))

    def test_linear_retract_can_cross_a_turn_boundary_without_a_full_rotation(self):
        controller = self.controller(joint_limits=[(-360, 360)] * 6)
        self.sdk.joints = FakeJoint(j6=-193.8)
        self.sdk.point.cfg = (0, 0, -1, 0)
        target = (310, 35, 455, -15, 25, 40)
        with patch.object(self.sdk, 'cnvrt_j', side_effect=[FakeJoint(j6=-533.6), FakeJoint(j6=-173.6)]), \
                patch.object(self.sdk, 'cnvrt_p', return_value=FakePoint(target, cfg=(0, 0, 0, 0))):
            controller.move_linear(euler2quat(target))
        self.assertEqual(self.last_motion('movl')[1].cfg, (0, 0, 0, 0))
        self.assertAlmostEqual(controller.commanded_joint_angles[5], -173.6)

    def test_linear_preflight_accepts_80_degree_change_and_rejects_a_real_joint_limit(self):
        controller = self.controller(joint_limits=[(-360, 360)] * 6)
        target = euler2quat((310, 35, 455, -15, 25, 40))
        self.sdk.joints = FakeJoint(j6=-100)
        self.sdk.ik_joints = FakeJoint(j6=-180)
        controller.check_linear_targets([target])

        self.sdk.joints = FakeJoint(j6=346)
        self.sdk.ik_joints = FakeJoint(j6=366)
        with self.assertRaisesRegex(TopstarMotionError, 'preflight target 1: .*J6=366'):
            controller.check_linear_targets([target])
        with self.assertRaisesRegex(TopstarMotionError, 'J6=366'):
            controller.move_linear(target)
        self.assertFalse(self.sdk.named('movl'))

    def test_circular_passes_distinct_via_end_and_additional_data_in_fifth_slot(self):
        controller = self.controller()
        via = euler2quat((305, 30, 450, 5, 15, 20))
        end = euler2quat((310, 25, 450, 10, 20, 30))
        controller.move_circular(via, end)
        args = self.last_motion('movc')
        assert_allclose(args[1].pose.tolist()[:3], (305, 30, 450))
        assert_allclose(args[2].pose.tolist()[:3], (310, 25, 450))
        self.assertTrue(args[3] is None or args[3] == 0)
        self.assertIsInstance(args[4], FakeMovPointAdd)
        assert_allclose(controller.commanded_pose, end, atol=1e-10)

    def test_orientation_roundtrip_at_gimbal_lock_is_compared_as_rotation(self):
        controller = self.controller()
        for angles in ((90, 0, 0), (0, 90, 0), (0, 0, 90), (40, 90, -20), (170, -89.99, -175)):
            with self.subTest(angles=angles):
                target = euler2quat((310, 35, 455, *angles))
                controller.move_linear(target)
                payload = self.last_motion('movl')[1].pose.tolist()[:6]
                assert_allclose(quat2mat(euler2quat(payload)), quat2mat(target), atol=1e-9)

    def test_trajectory_percentages_are_independent(self):
        controller = self.controller(command_speed=7, command_acceleration=2, command_deceleration=3)
        robot = SyncRobot(controller)
        for operation in (
            lambda: robot.move_joints([0] * 6),
            lambda: robot.move_linear([300, 25, 450, 10, 20, 30]),
            lambda: robot.move_circular([301, 26, 450, 10, 20, 30],
                                       [302, 25, 450, 10, 20, 30]),
        ):
            operation()
        for motion in (self.last_motion('movj')[2], self.last_motion('movl')[2],
                       self.last_motion('movc')[4]):
            self.assertEqual((motion.vel, motion.acc, motion.dec), (7, 2, 3))
        self.assertFalse(self.sdk.named('set_speed'))
        robot.command_speed = 9
        robot.move_joints([0] * 6)
        motion = self.last_motion('movj')[2]
        self.assertEqual((motion.vel, motion.acc, motion.dec), (9, 2, 3))
        self.assertFalse(self.sdk.named('set_speed'))

    def test_physical_speed_access_fails_explicitly(self):
        controller = self.controller()
        for field in ('linear_speed', 'angular_speed'):
            with self.subTest(field=field), self.assertRaises(NotImplementedError):
                setattr(controller, field, 5)
            with self.subTest(field=field), self.assertRaises(NotImplementedError):
                getattr(controller, field)

    def test_invalid_trajectory_percentage_fails_before_connection(self):
        for name in ('command_speed', 'command_acceleration', 'command_deceleration'):
            for value in (0, 101, 5.5, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.controller(**{name: value})
        self.assertFalse(self.sdk.named('connect'))

    def test_disable_latches_motion_until_prepare(self):
        controller = self.controller()
        controller.disable_servo()
        self.assertEqual(self.sdk.state.enable, 0)
        with self.assertRaises(TopstarStateError):
            controller.move_joints([0] * 6)
        controller.prepare_motion()
        self.assertEqual(self.sdk.state.enable, 1)

    def test_disable_works_despite_alarm_and_reports_failure(self):
        controller = self.controller()
        self.sdk.state.alarm = 1
        controller.disable_servo()
        self.assertEqual(self.sdk.state.enable, 0)
        self.sdk.fail["enable_servo"] = RuntimeError("off failed")
        with self.assertRaisesRegex(TopstarSDKError, "off failed"):
            controller.disable_servo()

    def test_blend_uses_explicit_lookup_and_zero_is_fine(self):
        controller = self.controller(blend_mapping={5: 20, 10: 50})
        controller.blend_radius = 5
        controller.move_joints([0] * 6)
        self.assertEqual(self.last_motion('movj')[2].cnt, 20)
        controller.blend_radius = 0
        controller.move_joints([0] * 6)
        self.assertEqual(self.last_motion('movj')[2].cnt, -1)
        with self.assertRaises((ValueError, TopstarError)):
            controller.blend_radius = 6

    def test_blend_without_calibration_rejects_nonzero(self):
        controller = self.controller()
        with self.assertRaises((ValueError, TopstarError)):
            controller.blend_radius = 1
        with self.assertRaises((ValueError, TopstarError)):
            controller.blend_radius = -1

    def test_joint_validation_and_optional_limits_precede_submission(self):
        controller = self.controller(joint_limits=[(-180, 180)] * 6)
        for value in ([0] * 5, [0] * 7, [0, 0, 0, 0, 0, math.nan], [181, 0, 0, 0, 0, 0]):
            with self.subTest(value=value), self.assertRaises((ValueError, TopstarError)):
                controller.move_joints(value)
        self.assertFalse(self.sdk.named('movj'))

    def test_pose_validation_rejects_nonfinite_zero_quaternion_and_wrong_size(self):
        controller = self.controller()
        for value in ((0,) * 6, (0,) * 7, (0, 0, math.inf, 1, 0, 0, 0)):
            with self.subTest(value=value), self.assertRaises((ValueError, TopstarError)):
                controller.move_linear(value)
        self.assertFalse(self.sdk.named('movl'))

    def test_six_axis_robot_rejects_elbow_parameter(self):
        controller = self.controller()
        pose = euler2quat((300, 25, 450, 10, 20, 30))
        with self.assertRaises((ValueError, TopstarError, NotImplementedError)):
            controller.move_linear(pose, elbow=10)
        with self.assertRaises((ValueError, TopstarError, NotImplementedError)):
            controller.move_circular(pose, pose, elbow=10)
        self.assertFalse(self.sdk.named('movl'))
        self.assertFalse(self.sdk.named('movc'))

    def test_motion_requires_local_enabled_supported_idle_nonrgm_state(self):
        controller = self.controller()
        for field, value in (('link', 0), ('enable', 0), ('alarm', 1), ('mode', 2),
                             ('remote', 1), ('cmd_queue', 1), ('cmd_queue', 2),
                             ('state', 1), ('state', 2), ('in_pos', 0), ('rgm_state', 1)):
            original = getattr(self.sdk.state, field)
            setattr(self.sdk.state, field, value)
            try:
                with self.subTest(field=field, value=value), self.assertRaises(TopstarError):
                    controller.move_joints([0] * 6)
            finally:
                setattr(self.sdk.state, field, original)
        self.assertFalse(self.sdk.named('movj'))

    def test_prepare_motion_explicitly_changes_mode_and_enables(self):
        self.sdk.state.mode = 3
        self.sdk.state.enable = 0
        controller = self.controller()
        controller.prepare_motion()
        self.assertEqual(self.sdk.state.mode, 100)
        self.assertEqual(self.sdk.state.enable, 1)
        self.assertTrue(self.sdk.named('enable_servo'))
        self.assertTrue(self.sdk.named('set_system_mode'))
        self.assertFalse(self.sdk.named('movj'))

    def test_prepare_refuses_alarm_and_remote_without_resetting_them(self):
        controller = self.controller()
        self.sdk.state.alarm = 1
        with self.assertRaises(TopstarError):
            controller.prepare_motion()
        self.sdk.state.alarm = 0
        self.sdk.state.remote = 1
        with self.assertRaises(TopstarError):
            controller.prepare_motion()
        self.assertFalse(self.sdk.named('enable_servo'))
        self.assertFalse(self.sdk.named('set_system_mode'))

    def test_abort_during_prepare_does_not_clear_stop_latch(self):
        self.sdk.state.mode = 3
        self.sdk.state.enable = 0
        controller = self.controller()
        self.sdk.mode_callback = controller.abort
        with self.assertRaises(TopstarMotionError):
            controller.prepare_motion()
        self.sdk.default_move_state = 1
        with self.assertRaises(TopstarStateError):
            controller.move_joints([0] * 6)
        self.assertFalse(self.sdk.named('movj'))
        self.assertFalse(self.sdk.named('enable_servo'))

    def test_motion_waits_for_completion_and_uses_finite_sdk_timeout(self):
        controller = self.controller(move_timeout=.5)
        self.sdk.move_states = [0, 0, 1]
        controller.move_joints([0] * 6)
        self.assertGreaterEqual(len(self.sdk.named('get_cur_move_state')), 3)
        waits = self.sdk.named('wait_move_done')
        self.assertTrue(waits)
        for _, args, kwargs in waits:
            timeout = kwargs.get('timeout_ms', args[1] if len(args) > 1 else -1)
            self.assertEqual(timeout, 100)

    def test_sdk_confirmation_timeout_retries_without_resubmitting_motion(self):
        controller = self.controller(move_timeout=.5)
        error = RuntimeError('命令超时')
        error.error_code = -98

        def first_wait_times_out():
            if len(self.sdk.named('wait_move_done')) == 1:
                raise error

        self.sdk.wait_callback = first_wait_times_out
        controller.move_joints([0] * 6)
        self.assertTrue(controller.last_motion_completed)
        self.assertEqual(len(self.sdk.named('wait_move_done')), 2)
        self.assertEqual(len(self.sdk.named('movj')), 1)
        self.assertFalse(self.sdk.named('abort'))

    def test_persistent_sdk_confirmation_timeout_aborts_at_motion_deadline(self):
        controller = self.controller(move_timeout=.05)
        error = RuntimeError('命令超时')
        error.error_code = -98
        self.sdk.fail['wait_move_done'] = error
        with self.assertRaises(TopstarTimeoutError):
            controller.move_joints([0] * 6)
        self.assertFalse(controller.last_motion_completed)
        self.assertEqual(len(self.sdk.named('movj')), 1)
        self.assertTrue(self.sdk.named('abort'))

    def test_missing_command_is_failure_not_success(self):
        controller = self.controller()
        self.sdk.default_move_state = 2
        with self.assertRaises(TopstarMotionError):
            controller.move_joints([0] * 6)

    def test_alarm_after_submission_aborts_before_reading_details_once(self):
        controller = self.controller()
        self.sdk.alarm_info = [
            {'level': 0, 'type': '2', 'code': '201', 'content': '系统急停',
             'reason': 'NULL', 'solution': '检查电气线路解除急停。'},
            {'code': '9001', 'content': 'mock alarm one'},
            {'code': '9002', 'content': 'mock alarm two'},
        ]
        self.sdk.post_submission_state = {'alarm': len(self.sdk.alarm_info)}
        with self.assertRaisesRegex(TopstarStateError, 'X5 reports 3 alarm') as caught:
            controller.move_linear(euler2quat((310, 25, 450, 10, 20, 30)))
        self.assertEqual(caught.exception.alarm_count, 3)
        for alarm in self.sdk.alarm_info:
            self.assertIn(alarm['code'], str(caught.exception))
            self.assertIn(alarm['content'], str(caught.exception))
        relevant = [name for name, _, _ in self.sdk.calls
                    if name in ('movl', 'abort', 'get_system_alarm_info')]
        self.assertEqual(relevant, ['movl', 'abort', 'get_system_alarm_info'])
        self.assertFalse(controller.last_motion_completed)

    def test_alarm_read_failure_preserves_the_original_motion_error(self):
        controller = self.controller()
        self.sdk.post_submission_state = {'alarm': 3}
        self.sdk.fail['get_system_alarm_info'] = RuntimeError('alarm query failed')
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always', RuntimeWarning)
            with self.assertRaisesRegex(TopstarStateError, 'X5 reports 3 alarm') as caught:
                controller.move_linear(euler2quat((310, 25, 450, 10, 20, 30)))
        self.assertEqual(caught.exception.alarm_count, 3)
        diagnostics = [str(caught.exception)] + [str(warning.message) for warning in emitted]
        self.assertIn('alarm query failed', ' '.join(diagnostics))
        relevant = [name for name, _, _ in self.sdk.calls
                    if name in ('movl', 'abort', 'get_system_alarm_info')]
        self.assertEqual(relevant, ['movl', 'abort', 'get_system_alarm_info'])

    def test_command_completion_does_not_override_axes_still_moving(self):
        controller = self.controller(move_timeout=.005)
        self.sdk.post_submission_state = {'in_pos': 0}
        with self.assertRaises(TopstarError):
            controller.move_joints([0] * 6)
        self.assertTrue(self.sdk.named('abort') or self.sdk.named('stop'))

    def test_late_completion_report_cannot_bypass_motion_deadline(self):
        controller = self.controller(move_timeout=.05)
        now = [100.0]
        self.sdk.move_state_callback = lambda: now.__setitem__(0, now[0] + .06)
        with patch('cri.topstar.topstar_controller.time.monotonic', side_effect=lambda: now[0]):
            with self.assertRaises(TopstarTimeoutError):
                controller.move_joints([0] * 6)
        self.assertFalse(controller.last_motion_completed)
        self.assertTrue(self.sdk.named('abort'))

    def test_completion_confirmation_cannot_bypass_motion_deadline(self):
        controller = self.controller(move_timeout=.05)
        now = [100.0]
        self.sdk.wait_callback = lambda: now.__setitem__(0, now[0] + .06)
        with patch('cri.topstar.topstar_controller.time.monotonic', side_effect=lambda: now[0]):
            with self.assertRaises(TopstarTimeoutError):
                controller.move_joints([0] * 6)
        self.assertFalse(controller.last_motion_completed)
        self.assertTrue(self.sdk.named('abort'))

    def test_motion_timeout_aborts_and_latches_until_prepare(self):
        controller = self.controller(move_timeout=.005)
        self.sdk.default_move_state = 0
        with self.assertRaises(TopstarTimeoutError):
            controller.move_joints([0] * 6)
        self.assertTrue(self.sdk.named('abort') or self.sdk.named('stop'))
        count = len(self.sdk.named('movj'))
        with self.assertRaises(TopstarError):
            controller.move_joints([0] * 6)
        self.assertEqual(len(self.sdk.named('movj')), count)
        self.sdk.default_move_state = 1
        controller.prepare_motion()
        controller.move_joints([0] * 6)

    def test_sdk_feedback_error_is_wrapped_with_original_cause(self):
        controller = self.controller()
        error = RuntimeError('connection reset')
        self.sdk.fail['get_cjoint'] = error
        with self.assertRaises(TopstarSDKError) as captured:
            _ = controller.joint_angles
        self.assertIs(captured.exception.__cause__, error)

    def test_vendor_diagnostics_are_retained_in_sdk_exception(self):
        controller = self.controller()
        error = RuntimeError('vendor communication failure')
        error.error_code = 123
        error.error_symbol = 'LINK_ERROR'
        error.error_message = 'Connection lost'
        self.sdk.fail['get_cjoint'] = error
        with self.assertRaises(TopstarSDKError) as captured:
            _ = controller.joint_angles
        self.assertEqual(captured.exception.error_code, 123)
        self.assertEqual(captured.exception.error_symbol, 'LINK_ERROR')
        self.assertEqual(captured.exception.error_message, 'Connection lost')

    def test_failed_submission_invalidates_commanded_target(self):
        controller = self.controller()
        self.sdk.fail['movj'] = RuntimeError('rejected')
        with self.assertRaises(TopstarError):
            controller.move_joints([10] * 6)
        with self.assertRaises(TopstarStateError):
            _ = controller.commanded_joint_angles
        with self.assertRaises(TopstarStateError):
            _ = controller.commanded_pose

    def test_ik_failure_does_not_submit_cartesian_motion(self):
        controller = self.controller()
        self.sdk.fail['cnvrt_j'] = RuntimeError('unreachable')
        with self.assertRaises(TopstarError):
            controller.move_linear(euler2quat((310, 25, 450, 10, 20, 30)))
        self.assertFalse(self.sdk.named('movl'))

    def test_stop_and_abort_prevent_subsequent_motion_until_explicit_prepare(self):
        controller = self.controller()
        for method in (controller.stop, controller.abort):
            method()
            with self.assertRaises(TopstarError):
                controller.move_joints([0] * 6)
            self.sdk.default_move_state = 1
            controller.prepare_motion()
            controller.move_joints([0] * 6)

    def test_stop_and_abort_are_flushed_before_idle_confirmation(self):
        controller = self.controller()
        for method in (controller.stop, controller.abort):
            self.sdk.calls.clear()
            method()
            names = [call[0] for call in self.sdk.calls]
            self.assertEqual(names[:2], [method.__name__, 'wait_cmd_send_done'])
            self.assertIn('get_all_system_state', names[2:])
            flush = self.sdk.named('wait_cmd_send_done')[-1]
            self.assertEqual(flush[1], (self.sdk.handle,))
            self.assertEqual(flush[2], {})

    def test_stop_from_another_thread_interrupts_blocking_motion(self):
        controller = self.controller(move_timeout=1)
        self.sdk.default_move_state = 0
        failures = []

        def move():
            try:
                controller.move_joints([0] * 6)
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=move, daemon=True)
        worker.start()
        self.assertTrue(self.sdk.submitted.wait(timeout=1), 'motion was not submitted')
        controller.stop()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive(), 'blocking motion did not react to stop')
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], TopstarMotionError)
        with self.assertRaises(TopstarStateError):
            controller.move_joints([0] * 6)

    def test_sync_robot_uses_existing_software_reference_frame(self):
        controller = self.controller()
        robot = SyncRobot(controller)
        robot.coord_frame = (100, 0, 0, 0, 0, 0)
        robot.move_linear((210, 25, 450, 10, 20, 30))
        payload = self.last_motion('movl')[1]
        assert_allclose(payload.pose.tolist()[:3], (310, 25, 450), atol=1e-10)
        assert_allclose(robot.target_pose, (210, 25, 450, 10, 20, 30), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
