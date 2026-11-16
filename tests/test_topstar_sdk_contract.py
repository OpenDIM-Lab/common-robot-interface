"""Optional checks against the real X5 Python SDK; never opens a connection.

Run with CPython 3.10 and the vendor's xapi 3.2.5 wheel installed.  The regular
hardware-free test environment skips this module when that SDK is unavailable.
These tests exercise Python/ctypes ABI contracts, not firmware compatibility.
"""

import ctypes
import inspect
import math

import numpy as np
import pytest


x5 = pytest.importorskip("xapi.api", reason="optional vendor xapi wheel not installed")


@pytest.mark.parametrize(
    "name",
    (
        "connect", "disconnect", "get_version", "get_all_system_state",
        "get_cjoint", "get_wpoint", "get_cpoint", "get_tf", "set_tf",
        "get_tfno", "set_tfno", "set_system_mode", "enable_servo",
        "movj", "movl", "movc", "servol", "servoj", "get_cur_move_state",
        "wait_move_done", "abort", "stop", "wait_cmd_send_done",
        "cnvrt_j", "cnvrt_p", "cnvrt_p_to_p",
    ),
)
def test_required_sdk_entry_points_exist(name):
    # Looking up attributes is intentional: __all__ and the supplied .pyi files
    # lag the compiled 3.2.5 module for some of these public functions.
    assert callable(getattr(x5, name, None)), name


def test_connection_and_wait_signatures_match_runtime_not_manual_typo():
    # bind() validates arguments without executing a vendor function.
    connection = inspect.signature(x5.connect)
    connection.bind(ip="192.0.2.1", link_timeout=2, retry_num=5)
    assert connection.parameters["link_timeout"].default == 2
    assert connection.parameters["retry_num"].default == 5
    waiting = inspect.signature(x5.wait_move_done)
    waiting.bind(handle=0, timeout_ms=1000)
    assert waiting.parameters["timeout_ms"].default == -1


def test_motion_keyword_signatures_bind_without_issuing_motion():
    point = x5.Point(pose=x5.Pose(), uf=0, tf=1, cfg=(0, 0, 0, 1))
    options = x5.MovPointAdd(vel=10, cnt=-1)
    inspect.signature(x5.movj).bind(handle=0, point=x5.Joint(), additional_data=options)
    inspect.signature(x5.movl).bind(handle=0, point=point, additional_data=options)
    inspect.signature(x5.movc).bind(
        handle=0, middle_point=point, target_point=point,
        central_angle=0, additional_data=options,
    )
    for name, target in (("servol", point), ("servoj", x5.Joint())):
        inspect.signature(getattr(x5, name)).bind(
            handle=0, **{"point" if name == "servol" else "joint": target},
            cmdt=0.032, lookahead=0, gain=5, vel=10, acc=10,
        )


def test_pose_and_joint_keyword_constructors_and_ctypes_fields():
    pose_values = dict(x=10, y=20, z=30, a=15, b=-25, c=35, e1=0, e2=0, e3=0)
    joint_values = dict(j1=1, j2=-2, j3=3, j4=-4, j5=5, j6=-6, e1=0, e2=0, e3=0)
    for cls, values in ((x5.Pose, pose_values), (x5.Joint, joint_values)):
        value = cls(**values)
        assert isinstance(value, ctypes.Structure)
        assert cls._fields_ == [(name, ctypes.c_double) for name in values]
        assert [getattr(value, name) for name in values] == list(values.values())
        assert ctypes.sizeof(value) == 9 * ctypes.sizeof(ctypes.c_double)


def test_cartesian_point_preserves_frame_ids_and_configuration():
    pose = x5.Pose(x=100, y=20, z=300, a=10, b=20, c=30)
    point = x5.Point(pose=pose, uf=0, tf=3, cfg=(-1, 0, 1, 4))
    fields = dict(x5.Point._fields_)
    assert len(fields) == 4
    assert fields["pose"] is x5.Pose
    assert fields["uf"] is ctypes.c_int
    assert fields["tf"] is ctypes.c_int
    assert fields["cfg"]._type_ is ctypes.c_int
    assert fields["cfg"]._length_ == 4
    assert (point.uf, point.tf) == (0, 3)
    assert tuple(point.cfg) == (-1, 0, 1, 4)
    assert (point.pose.x, point.pose.c) == (100, 30)


def test_movement_options_use_fine_and_disable_compensations():
    options = x5.MovPointAdd(vel=10, cnt=-1, acc=25, dec=30)
    integer_fields = (
        "vel", "cnt", "acc", "dec", "offset", "toffset", "vision_num",
        "track_num", "wrist", "inc", "search", "rtcp",
    )
    fields = dict(x5.MovPointAdd._fields_)
    assert len(fields) == 14
    assert all(fields[name] is ctypes.c_int for name in integer_fields)
    assert fields["offset_data"] is x5.Pose
    assert fields["toffset_data"] is x5.Pose
    assert (options.vel, options.cnt, options.acc, options.dec) == (10, -1, 25, 30)
    assert (options.offset, options.toffset) == (-2, -2)
    assert (options.vision_num, options.track_num) == (-1, -1)
    assert (options.wrist, options.inc, options.search, options.rtcp) == (0, 0, 0, 0)


def test_all_system_state_exposes_full_queue_and_nested_data():
    fields = dict(x5.AllSystemState._fields_)
    assert len(fields) == 18
    for name in (
        "link", "enable", "alarm", "mode", "state", "in_pos", "remote",
        "cmd_queue", "speed", "uf_no", "tf_no", "rgm_state",
    ):
        assert fields[name] is ctypes.c_int
    assert fields["user_point"] is x5.Point
    assert fields["world_point"] is x5.Point
    assert fields["joint"] is x5.Joint
    for name in ("di_list", "do_list"):
        assert fields[name]._type_ is ctypes.c_ubyte
        assert fields[name]._length_ == 32
    assert fields["ei_list"]._type_ is ctypes.c_int
    assert fields["ei_list"]._length_ == 8


def test_runtime_mode_constants_match_documented_values():
    assert x5.SYSTEM_MODE_AUTO == 2
    assert x5.SYSTEM_MODE_MANUAL == 3
    assert x5.SYSTEM_MODE_DEBUG == 5
    assert x5.SYSTEM_MODE_AUTOCMD == 100


def test_conversion_helpers_accept_real_sdk_pose_and_use_wxyz_degrees():
    from cri.topstar.topstar_controller import x5_pose_to_quat, quat_to_x5_pose

    pose = x5.Pose(x=123, y=-45, z=678, a=90, b=0, c=0)
    quaternion_pose = x5_pose_to_quat(pose, "sxyz")
    np.testing.assert_allclose(
        quaternion_pose,
        (123, -45, 678, math.sqrt(0.5), math.sqrt(0.5), 0, 0),
        atol=1e-12,
    )
    restored = quat_to_x5_pose(quaternion_pose, "sxyz", x5)
    assert isinstance(restored, x5.Pose)
    np.testing.assert_allclose(
        [restored.x, restored.y, restored.z, restored.a, restored.b, restored.c],
        [123, -45, 678, 90, 0, 0], atol=1e-12,
    )
    assert (restored.e1, restored.e2, restored.e3) == (0, 0, 0)


@pytest.mark.parametrize("axes", ("sxyz", "rxyz", "szyx", "rzyx"))
def test_real_sdk_pose_roundtrip_respects_explicit_rotation_convention(axes):
    from cri.topstar.topstar_controller import x5_pose_to_quat, quat_to_x5_pose

    original = x5.Pose(x=101.5, y=-22.25, z=330, a=23, b=-31, c=47)
    quaternion_pose = x5_pose_to_quat(original, axes)
    restored = quat_to_x5_pose(quaternion_pose, axes, x5)
    np.testing.assert_allclose(
        [restored.x, restored.y, restored.z, restored.a, restored.b, restored.c],
        [original.x, original.y, original.z, original.a, original.b, original.c],
        atol=1e-10,
    )


def test_nonunit_quaternion_is_normalized_before_real_sdk_pose_construction():
    from cri.topstar.topstar_controller import quat_to_x5_pose

    restored = quat_to_x5_pose((10, 20, 30, 2, 0, 0, 0), "sxyz", x5)
    np.testing.assert_allclose(
        [restored.x, restored.y, restored.z, restored.a, restored.b, restored.c],
        [10, 20, 30, 0, 0, 0], atol=1e-12,
    )
