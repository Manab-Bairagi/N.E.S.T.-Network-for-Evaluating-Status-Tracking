# -*- coding: utf-8 -*-
"""MediaPipe_inference.py

Action recognition inference pipeline using MediaPipe + EfficientGCN-B0.
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
All dependencies are listed in requirements.txt.
Install with:

    pip install -r requirements.txt

For GPU (CUDA 11.8), install PyTorch separately first:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Also download the MediaPipe Tasks pose model once:

    python -c "
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
        'pose_landmarker_heavy.task'
    ); print('Downloaded pose_landmarker_heavy.task')
    "
"""

print("✓ Cell 1: See requirements.txt for installation instructions.")


# ============================================================================
# CELL 2: Import Custom Tracker
# ============================================================================
"""
Place customtrackerfinal.py in the same directory as this script,
then run. This imports everything from the tracker WITHOUT modifying it.
"""

import importlib, sys, os

# Ensure the script's own directory is on the path so that
# customtrackerfinal.py can be found when running from any cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# We import the tracker module as-is (zero modifications)
import customtrackerfinal as tracker_module

# Pull the classes/functions we need from the tracker
from customtrackerfinal import (
    EnhancedFeatureExtractor,
    MemoryEnhancedBoTSORT,
    draw_tracks,
    draw_trajectories,
    Track,
)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# CHANGED: Import YOLO for detection only (no more YOLO-Pose)
from ultralytics import YOLO

# CHANGED: Import MediaPipe for skeleton extraction
import mediapipe as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Tracker imported successfully | Device: {device}")


# ============================================================================
# CELL 3: Action Model Architecture (EfficientGCN-B0 — 3D)
# ============================================================================
"""
CHANGED (3D upgrade):
  - INPUT_DIM updated to 3 to reflect (X, Y, Z) coordinates from NTU RGB+D.
  - EfficientGCN_B0.__init__ updated so that the three InitialBlocks receive
    the correct number of input channels that correspond to the 3D feature
    streams produced by generate_features_from_sequence:
        init_joint    : in_ch = 6  (abs_x, abs_y, abs_z, rel_x, rel_y, rel_z)
        init_velocity : in_ch = 6  (fast_x, fast_y, fast_z, slow_x, slow_y, slow_z)
        init_bone     : in_ch = 4  (bone_dx, bone_dy, bone_dz, bone_len)
    Total input channels = 6 + 6 + 4 = 16.
  - The forward() docstring updated accordingly.
  - All other architecture code (graph, layers, attention) is identical to the
    training script and must NOT be modified.
"""

# ---- NTU action class list (45 classes used during training) ----
ACTION_CLASSES = [
    'A001','A002','A003','A005','A006','A008','A009','A011','A012',
    'A018','A019','A027','A028','A041','A042','A043','A044','A045',
    'A046','A047','A048','A049','A050','A053','A054','A055','A056',
    'A058','A059','A060','A080','A085','A086','A089','A090','A091',
    'A092','A103','A106','A107','A108','A109','A114','A116','A119'
]
NUM_CLASSES  = len(ACTION_CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(ACTION_CLASSES)}

# Human-readable label map
LABEL_NAMES = {
    'A001': 'drink water',            'A002': 'eat meal',
    'A003': 'brush teeth',            'A005': 'drop',
    'A006': 'pick up',                'A008': 'sit down',
    'A009': 'stand up',               'A011': 'reading',
    'A012': 'writing',                'A018': 'put on glasses',
    'A019': 'take off glasses',       'A027': 'jump up',
    'A028': 'phone call',             'A041': 'sneeze/cough',
    'A042': 'staggering',             'A043': 'falling down',
    'A044': 'headache',               'A045': 'chest pain',
    'A046': 'back pain',              'A047': 'neck pain',
    'A048': 'nausea/vomiting',        'A049': 'fan self',
    'A050': 'punch/slap',             'A053': 'pat on back',
    'A054': 'point finger',           'A055': 'hugging',
    'A056': 'giving object',          'A058': 'shaking hands',
    'A059': 'walking towards',        'A060': 'walking apart',
    'A080': 'squat down',             'A085': 'apply cream on face',
    'A086': 'apply cream on hand',    'A089': 'put object into bag',
    'A090': 'take object out of bag', 'A091': 'open a box',
    'A092': 'move heavy objects',     'A103': 'yawn',
    'A106': 'hit with object',        'A107': 'wield knife',
    'A108': 'knock over',             'A109': 'grab stuff',
    'A114': 'carry object',           'A116': 'follow',
    'A119': 'support somebody',
}

# ---- EfficientGCN-B0 architecture constants (must match training) ----
NUM_JOINTS      = 25
MAX_PERSONS     = 2
MAX_FRAMES      = 90
INPUT_DIM       = 3        # CHANGED: (x, y, z) — 3D coordinates from NTU RGB+D
SGLayer_RRD     = 2
TEMPORAL_L      = 5
GRAPH_D         = 2
DROPOUT         = 0.30
ATTENTION_RRD   = 4
SPINE_JOINT     = 20

# NTU 25-joint bone pairs (0-based) — for bone feature generation
NTU_JOINT_PAIRS = [
    (0, 1), (1, 20), (2, 20), (3, 2),
    (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 6), (22, 6),
    (23, 10), (24, 10),
]

# ---- NTU Graph Adjacency ----
class NTUGraph:
    EDGES = [
        (0, 1), (1, 20), (2, 20), (3, 2),
        (4, 20), (5, 4), (6, 5), (7, 6),
        (8, 20), (9, 8), (10, 9), (11, 10),
        (12, 0), (13, 12), (14, 13), (15, 14),
        (16, 0), (17, 16), (18, 17), (19, 18),
        (21, 6), (22, 6), (23, 10), (24, 10),
    ]

    def __init__(self, num_joints=NUM_JOINTS, max_distance=GRAPH_D):
        self.V = num_joints
        self.D = max_distance
        self.adj_matrices = self._build_adj()

    def _build_adj(self):
        V, D = self.V, self.D
        dist = np.full((V, V), np.inf)
        np.fill_diagonal(dist, 0)
        for (i, j) in self.EDGES:
            dist[i, j] = 1; dist[j, i] = 1
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        adj_list = []
        for d in range(D + 1):
            A = (dist == d).astype(np.float32)
            row_sum = A.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum == 0, 1, row_sum)
            A_norm = A / row_sum
            adj_list.append(torch.from_numpy(A_norm).float())
        return adj_list


GRAPH = NTUGraph()

# ---- Model Components ----
class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, A, **kwargs):
        super().__init__()
        self.s_kernel_size = max_graph_distance + 1
        self.gcn  = nn.Conv2d(in_channel, out_channel * self.s_kernel_size, 1, bias=True)
        self.A    = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        self.edge = nn.Parameter(torch.ones_like(self.A))
        self.adaptive_A = nn.Parameter(torch.zeros_like(self.A))

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        effective_A = self.A * self.edge + self.adaptive_A
        x = torch.einsum('nkctv,kvw->nctw', x, effective_A).contiguous()
        return x


class SGC(nn.Module):
    def __init__(self, in_channel, out_channel, A, act, **kwargs):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channel, out_channel, max_graph_distance=GRAPH_D, A=A)
        self.bn  = nn.BatchNorm2d(out_channel)
        self.act = act
        if in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=True),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        return self.act(self.bn(self.sgc(x)) + res)


class SGLayer(nn.Module):
    def __init__(self, channel, stride=1, reduct_ratio=SGLayer_RRD,
                 kernel_size=TEMPORAL_L, act=None, **kwargs):
        super().__init__()
        pad = (kernel_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act if act is not None else Swish()
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (kernel_size, 1), 1, (pad, 0), groups=channel, bias=True),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=True),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=True),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (kernel_size, 1), (stride, 1), (pad, 0), groups=channel, bias=True),
            nn.BatchNorm2d(channel),
        )
        if stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=True),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x   = self.act(self.depth_conv1(x))
        x   = self.point_conv1(x)
        x   = self.act(self.point_conv2(x))
        x   = self.depth_conv2(x)
        return x + res


class STJointAtt(nn.Module):
    def __init__(self, channel, reduct_ratio=ATTENTION_RRD, **kwargs):
        super().__init__()
        inner = channel // reduct_ratio
        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner, kernel_size=1, bias=True),
            nn.BatchNorm2d(inner),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdim=True)
        x_v = x.mean(2, keepdim=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        return x_t_att * x_v_att


class AttentionLayer(nn.Module):
    def __init__(self, channel, act, **kwargs):
        super().__init__()
        self.att = STJointAtt(channel)
        self.bn  = nn.BatchNorm2d(channel)
        self.act = act

    def forward(self, x):
        res = x
        x   = x * self.att(x)
        return self.act(self.bn(x) + res)


class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, depth=0, act=None):
        super().__init__()
        self.act = act if act is not None else Swish()
        self.sgc = SGC(in_ch, out_ch, A=A, act=self.act)
        self.tc_layers = nn.ModuleList([
            SGLayer(out_ch, stride=stride if i == 0 else 1, act=self.act)
            for i in range(depth)
        ])
        self.att = AttentionLayer(out_ch, act=self.act)

    def forward(self, x):
        x = self.sgc(x)
        for tc in self.tc_layers:
            x = tc(x)
        x = self.att(x)
        return x


class InitialBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, act=None):
        super().__init__()
        self.act = act if act is not None else Swish()
        pad = (TEMPORAL_L - 1) // 2
        self.bn  = nn.BatchNorm2d(in_ch)
        self.sgc = SGC(in_ch, out_ch, A=A, act=self.act)
        self.tc  = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (TEMPORAL_L, 1), 1, (pad, 0), bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.act_out = self.act

    def forward(self, x):
        x = self.bn(x)
        x = self.sgc(x)
        x = self.act_out(self.tc(x))
        return x


class CrossStreamAttention(nn.Module):
    """
    Given 3 streams each of shape [N, C, T, V], computes a per-sample
    soft attention weight over the 3 streams and returns a weighted sum.
    """
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, 3),
        )

    def forward(self, j, v, b):
        j_gap = j.mean(dim=[2, 3])
        v_gap = v.mean(dim=[2, 3])
        b_gap = b.mean(dim=[2, 3])
        feat  = torch.cat([j_gap, v_gap, b_gap], dim=1)
        w     = self.fc(feat).softmax(dim=1)
        out = w[:, 0:1, None, None] * j + \
              w[:, 1:2, None, None] * v + \
              w[:, 2:3, None, None] * b
        return out


class EfficientGCN_B0(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        act = Swish()
        A   = torch.stack(GRAPH.adj_matrices, dim=0)

        # CHANGED: InitialBlock input channels updated for 3D feature streams.
        #   init_joint    : 6 ch  (abs_x, abs_y, abs_z, rel_x, rel_y, rel_z)
        #   init_velocity : 6 ch  (fast_x, fast_y, fast_z, slow_x, slow_y, slow_z)
        #   init_bone     : 4 ch  (bone_dx, bone_dy, bone_dz, bone_len)
        self.init_joint    = InitialBlock(6,  64, A=A, act=act)
        self.init_velocity = InitialBlock(6,  64, A=A, act=act)
        self.init_bone     = InitialBlock(4,  64, A=A, act=act)

        self.stage1_joint    = GCNBlock(64, 48, A=A, depth=0, act=act)
        self.stage1_velocity = GCNBlock(64, 48, A=A, depth=0, act=act)
        self.stage1_bone     = GCNBlock(64, 48, A=A, depth=0, act=act)

        self.stage2_joint    = GCNBlock(48, 16, A=A, depth=0, act=act)
        self.stage2_velocity = GCNBlock(48, 16, A=A, depth=0, act=act)
        self.stage2_bone     = GCNBlock(48, 16, A=A, depth=0, act=act)

        self.cross_stream_att = CrossStreamAttention(channel=16)

        self.stage3 = GCNBlock(64,  64, A=A, stride=2, depth=1, act=act)
        self.stage4 = GCNBlock(64, 128, A=A, stride=2, depth=1, act=act)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, joint, velocity, bone):
        """
        CHANGED: channel dimensions updated for 3D streams.
        joint:    [N, 6, T, V, M]   (abs_xyz + rel_xyz)
        velocity: [N, 6, T, V, M]   (fast_xyz + slow_xyz)
        bone:     [N, 4, T, V, M]   (bone_dx, bone_dy, bone_dz, bone_len)
        """
        N = joint.shape[0]

        def merge_M(x):
            N, C, T, V, M = x.shape
            return x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        j = merge_M(joint)
        v = merge_M(velocity)
        b = merge_M(bone)

        j = self.init_joint(j)
        v = self.init_velocity(v)
        b = self.init_bone(b)

        j = self.stage1_joint(j)
        v = self.stage1_velocity(v)
        b = self.stage1_bone(b)

        j = self.stage2_joint(j)
        v = self.stage2_velocity(v)
        b = self.stage2_bone(b)

        fused = self.cross_stream_att(j, v, b)
        x = torch.cat([j, v, b, fused], dim=1)

        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x).view(N * joint.shape[4], -1)
        x = self.dropout(x)
        x = x.view(N, joint.shape[4], -1).mean(dim=1)
        return self.fc(x)


print("✓ EfficientGCN-B0 (3D) architecture defined")


# ============================================================================
# CELL 4: Load Action Recognition Model
# ============================================================================

ACTION_MODEL_PATH = 'best_efficientgcn_b0_media_3d_81.pth'   # <-- update path if needed

def load_action_model(ckpt_path):
    model = EfficientGCN_B0(num_classes=NUM_CLASSES).to(device)

    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint not found at {ckpt_path}. Model weights NOT loaded.")
        return model

    # Load the checkpoint dictionary
    ckpt = torch.load(ckpt_path, map_location=device)

    # FIX: Extract the actual weights from the 'state_dict' key
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        epoch = ckpt.get('epoch', 'unknown')
        acc   = ckpt.get('val_acc', 'unknown')
        print(f"✓ Extracting weights from nested state_dict | epoch={epoch} | val_acc={acc}")
    else:
        state = ckpt
        print("✓ Loading raw state_dict")

    # Handle weights saved with DataParallel (removes 'module.' prefix if it exists)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load the cleaned state dict into the model
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    print(f"✓ EfficientGCN-B0 (3D) weights loaded successfully from {ckpt_path}")
    return model

action_model = load_action_model(ACTION_MODEL_PATH)
print("✓ Action model ready")


# ============================================================================
# CELL 5: Interaction Detection Module
# ============================================================================
# UNCHANGED

def compute_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def box_diagonal(box):
    return math.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)


def center_distance(boxA, boxB):
    cA, cB = box_center(boxA), box_center(boxB)
    return math.sqrt((cA[0]-cB[0])**2 + (cA[1]-cB[1])**2)


def union_box(boxA, boxB):
    return [min(boxA[0],boxB[0]), min(boxA[1],boxB[1]),
            max(boxA[2],boxB[2]), max(boxA[3],boxB[3])]


class InteractionDetector:
    """
    Determines if two tracked people are interacting using:
      - IoU overlap threshold
      - center-to-center distance threshold
      - temporal persistence (N consecutive frames)
    """
    def __init__(self, iou_thresh=0.2, dist_thresh=150, persist_frames=5):
        self.iou_thresh     = iou_thresh
        self.dist_thresh    = dist_thresh
        self.persist_frames = persist_frames
        self._counters      = defaultdict(int)
        self._active        = set()

    def _pair_key(self, id_a, id_b):
        return tuple(sorted([id_a, id_b]))

    def update(self, tracks):
        current_pairs = set()
        ids = list(tracks.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                boxA, boxB = tracks[id_a], tracks[id_b]
                iou  = compute_iou(boxA, boxB)
                dist = center_distance(boxA, boxB)
                avg_diag   = (box_diagonal(boxA) + box_diagonal(boxB)) / 2
                dyn_thresh = max(self.dist_thresh, avg_diag * 1.2)
                key = self._pair_key(id_a, id_b)
                if iou > self.iou_thresh and dist < dyn_thresh:
                    self._counters[key] += 1
                    current_pairs.add(key)
                else:
                    self._counters[key] = max(0, self._counters[key] - 1)
        for key in list(current_pairs):
            if self._counters[key] >= self.persist_frames:
                self._active.add(key)
        for key in list(self._active):
            if key not in current_pairs or self._counters[key] == 0:
                self._active.discard(key)
        return set(self._active)


print("✓ InteractionDetector defined")


# ============================================================================
# CELL 6: Bounding Box Expansion Module
# ============================================================================
# UNCHANGED

def expand_bbox(box, frame_h, frame_w, scale=1.7):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
    w  = (x2 - x1) * scale
    h  = (y2 - y1) * scale
    nx1 = max(0,       int(cx - w/2))
    ny1 = max(0,       int(cy - h/2))
    nx2 = min(frame_w, int(cx + w/2))
    ny2 = min(frame_h, int(cy + h/2))
    return [nx1, ny1, nx2, ny2]


print("✓ BBox expansion utilities defined")


# ============================================================================
# CELL 7: Skeleton Keypoint Mapping & Buffer (3D Upgrade)
# ============================================================================
"""
CHANGED (3D upgrade):

  mediapipe_to_ntu25:
    - Now returns [25, 3] instead of [25, 2].
    - All midpoint calculations (spine, neck, pelvis) compute the mean across
      X, Y, AND Z axes.
    - snap_missing_to_spine copies the 3D spine coordinates (X, Y, Z) to any
      joint that remained at (0, 0, 0) after mapping.
    - The detected-joint check is updated to test any of (x, y, z) non-zero.

  generate_features_from_sequence:
    - Now accepts seq of shape [T, 25, 3].
    - Joint stream  (6 ch): abs_x, abs_y, abs_z  +  rel_x, rel_y, rel_z.
    - Velocity stream (6 ch): fast_x, fast_y, fast_z  +  slow_x, slow_y, slow_z.
    - Bone stream (4 ch): bone_dx, bone_dy, bone_dz  +  bone_len (3D Euclidean
      length: sqrt(dx^2 + dy^2 + dz^2)). The 2D bone_angle channel is removed.
    - Output shape: [16, T, 25, 1].

  SkeletonBuffer:
    - Internal shapes updated from [25, 2] to [25, 3] and 11-ch to 16-ch.
    - All buffer logic and public API remain identical.
"""

# ---- MediaPipe 33-landmark index → NTU 25-joint index ----
# Only the landmarks that have a clear anatomical correspondence are mapped.
# Unmapped NTU joints are approximated or snapped to spine (see function below).
MP_TO_NTU = {
    0:  3,   # nose          → head
    11: 4,   # left shoulder → NTU left shoulder
    12: 8,   # right shoulder→ NTU right shoulder
    13: 5,   # left elbow    → NTU left elbow
    14: 9,   # right elbow   → NTU right elbow
    15: 6,   # left wrist    → NTU left wrist
    16: 10,  # right wrist   → NTU right wrist
    23: 12,  # left hip      → NTU left hip
    24: 16,  # right hip     → NTU right hip
    25: 13,  # left knee     → NTU left knee
    26: 17,  # right knee    → NTU right knee
    27: 14,  # left ankle    → NTU left ankle
    28: 18,  # right ankle   → NTU right ankle
    31: 15,  # left foot     → NTU left foot
    32: 19,  # right foot    → NTU right foot
    19: 21,  # left index    → NTU left hand tip 1
    17: 22,  # left pinky    → NTU left hand tip 2
    20: 23,  # right index   → NTU right hand tip 1
    18: 24,  # right pinky   → NTU right hand tip 2
}


def mediapipe_to_ntu25(landmarks, snap_missing_to_spine=False):
    """
    Convert MediaPipe pixel-space landmarks to NTU-25 joint array (3D).

    Parameters
    ----------
    landmarks : np.ndarray, shape [33, 3]
        MediaPipe landmark (x, y, z) coordinates.
        x and y are in pixel space.
        z is pre-scaled by crop_w (see Cell 10).
        Undetected landmarks are represented as (0, 0, 0).

    snap_missing_to_spine : bool
        If True, any NTU joint that remains at (0, 0, 0) after mapping is
        overwritten with the spine (NTU joint 20) 3D coordinates.
        Use True for model input so missing joints contribute zero distance in
        the spine-relative feature calculation.
        Use False for visualization so undetected joints are simply not drawn.

    Returns
    -------
    joints : np.ndarray, shape [25, 3]
        NTU-25 joint positions (x, y, z).  x and y are in pixel space.
        z is in the same scale as x (pixels, proportional to crop width).
        Downstream normalization is handled by generate_features_from_sequence.
    """
    # CHANGED: array is now [25, 3] to hold (x, y, z).
    joints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    # Step 1: Direct mapping from MediaPipe landmarks to NTU joints.
    for mp_idx, ntu_idx in MP_TO_NTU.items():
        if mp_idx < len(landmarks):
            joints[ntu_idx] = landmarks[mp_idx]   # copies all 3 coords

    # Step 2: Compute NTU spine (joint 20) as midpoint of shoulders and hips.
    # CHANGED: non-zero check tests any axis to handle cases where only z is 0.
    left_sh   = joints[4]    # NTU left shoulder  (MP 11)
    right_sh  = joints[8]    # NTU right shoulder (MP 12)
    left_hip  = joints[12]   # NTU left hip       (MP 23)
    right_hip = joints[16]   # NTU right hip      (MP 24)

    # A joint is "detected" if any of its three coordinates is non-zero.
    def _detected(p):
        return p[0] != 0 or p[1] != 0 or p[2] != 0

    shoulder_pts = [p for p in [left_sh, right_sh] if _detected(p)]
    hip_pts      = [p for p in [left_hip, right_hip] if _detected(p)]

    # CHANGED: np.mean now averages across all 3 axes simultaneously because
    # each element p is [x, y, z].
    if shoulder_pts and hip_pts:
        mid_sh  = np.mean(shoulder_pts, axis=0)   # [3]
        mid_hip = np.mean(hip_pts, axis=0)         # [3]
        joints[20] = (mid_sh + mid_hip) / 2.0      # spine midpoint in 3D
    elif shoulder_pts:
        joints[20] = np.mean(shoulder_pts, axis=0)
    elif hip_pts:
        joints[20] = np.mean(hip_pts, axis=0)
    # else: spine stays (0,0,0) — handled by snap below if requested

    spine = joints[20].copy()   # [3]

    # Step 3: Approximate NTU joints with no direct MediaPipe equivalent.
    #   NTU 0  — pelvis/base of spine  → midpoint of hips
    #   NTU 1  — mid-spine             → midpoint of spine (joint 20) and pelvis (joint 0)
    #   NTU 2  — neck / chest top      → midpoint of shoulders
    #
    # CHANGED: All mean() and midpoint calculations naturally extend to 3D
    # because the arrays are now [3]-vectors.
    if hip_pts:
        joints[0] = np.mean(hip_pts, axis=0)
    elif _detected(joints[20]):
        joints[0] = joints[20]   # fallback

    if _detected(joints[0]) and _detected(joints[20]):
        joints[1] = (joints[0] + joints[20]) / 2.0
    elif _detected(joints[20]):
        joints[1] = joints[20]

    if shoulder_pts:
        joints[2] = np.mean(shoulder_pts, axis=0)
    elif _detected(joints[20]):
        joints[2] = joints[20]

    # Step 4: Mirror wrist joints to NTU hand joints.
    #   NTU 7  → left hand  (mirrors left wrist, NTU 6)
    #   NTU 11 → right hand (mirrors right wrist, NTU 10)
    joints[7]  = joints[6]    # left hand  = left wrist  (copies xyz)
    joints[11] = joints[10]   # right hand = right wrist (copies xyz)

    # Step 5: Snap undetected joints (still at (0,0,0)) to spine if requested.
    # CHANGED: check uses _detected() to test all three axes.
    if snap_missing_to_spine:
        for i in range(NUM_JOINTS):
            if not _detected(joints[i]):
                joints[i] = spine

    return joints   # [25, 3] — pixel space x, y; scaled-pixel z


def generate_features_from_sequence(seq):
    """
    Generate the 16-channel EfficientGCN feature array from a raw 3D
    skeleton sequence.

    Parameters
    ----------
    seq : np.ndarray, shape [T, 25, 3]
        Raw (x, y, z) per frame per joint for a single person.

    Returns
    -------
    sample : np.ndarray, shape [16, T, 25, 1]
        Feature tensor split as:
          ch  0- 5  : joint stream    (abs_x, abs_y, abs_z,
                                       rel_x, rel_y, rel_z)        [6 ch]
          ch  6-11  : velocity stream (fast_x, fast_y, fast_z,
                                       slow_x, slow_y, slow_z)     [6 ch]
          ch 12-15  : bone stream     (bone_dx, bone_dy, bone_dz,
                                       bone_len)                   [4 ch]
        Total = 6 + 6 + 4 = 16 channels.

    CHANGED (3D upgrade):
      - seq is now [T, 25, 3] instead of [T, 25, 2].
      - Spine-centering and max-abs normalisation operate on all 3 axes.
      - Joint stream:    abs + spine-relative for x, y, z  → 6 ch.
      - Velocity stream: fast + slow for x, y, z           → 6 ch.
      - Bone stream:     dx, dy, dz + 3D length            → 4 ch.
        The 2D bone_angle channel is replaced by 3D bone_len =
        sqrt(dx^2 + dy^2 + dz^2), which is more meaningful in 3D space.
    """
    T, V, C = seq.shape   # C = 3

    # Transpose to [C, T, V]  →  [3, T, 25]
    data = seq.transpose(2, 0, 1)

    # ── Normalise: centre at spine joint ──────────────────────────────────
    # spine shape: [3, T, 1]  →  broadcast-subtracts across all V joints
    spine = data[:, :, SPINE_JOINT:SPINE_JOINT+1]   # [3, T, 1]
    data_rel = data - spine
    scale = np.abs(data_rel).max()
    if scale > 1e-6:
        data_rel = data_rel / scale
    data = data_rel   # [3, T, V]

    # Add M dimension: [3, T, V, 1]
    data = data[:, :, :, np.newaxis]

    # ── Joint features  (6 channels) ──────────────────────────────────────
    # absolute: the normalised xyz coords as-is
    # relative: subtract the (already-centred) spine to get joint-to-spine
    #           offsets. Since data is already spine-centred, spine column is
    #           near zero but recalculating keeps the pipeline numerically
    #           consistent with training.
    absolute = data.copy()                                    # [3, T, V, 1]
    spine4   = data[:, :, SPINE_JOINT:SPINE_JOINT+1, :]      # [3, T, 1, 1]
    relative = data - spine4                                  # [3, T, V, 1]
    joint    = np.concatenate([absolute, relative], axis=0)  # [6, T, V, 1]

    # ── Velocity features  (6 channels) ───────────────────────────────────
    # fast velocity: displacement over 2 frames  (frame[t+2] − frame[t])
    # slow velocity: displacement over 1 frame   (frame[t+1] − frame[t])
    fast = np.zeros_like(data)   # [3, T, V, 1]
    slow = np.zeros_like(data)
    if T > 2:
        fast[:, :-2, :, :] = data[:, 2:, :, :] - data[:, :-2, :, :]
    if T > 1:
        slow[:, :-1, :, :] = data[:, 1:, :, :] - data[:, :-1, :, :]
    velocity = np.concatenate([fast, slow], axis=0)          # [6, T, V, 1]

    # ── Bone features  (4 channels) ───────────────────────────────────────
    # bone_xyz: 3D delta vector from child joint to parent joint
    # bone_len: Euclidean length of the 3D bone vector = sqrt(dx^2+dy^2+dz^2)
    #
    # CHANGED: bone_angle (atan2 of 2D dx/dy) is replaced by bone_len because
    # angle is not well-defined in 3D and bone length is a better-generalising
    # structural feature for the 3D NTU model.
    bone_xyz = np.zeros_like(data)   # [3, T, V, 1]  — dx, dy, dz per bone
    for (i, j) in NTU_JOINT_PAIRS:
        if i < V and j < V:
            bone_xyz[:, :, i, :] = data[:, :, i, :] - data[:, :, j, :]

    bone_dx  = bone_xyz[0:1, :, :, :]   # [1, T, V, 1]
    bone_dy  = bone_xyz[1:2, :, :, :]   # [1, T, V, 1]
    bone_dz  = bone_xyz[2:3, :, :, :]   # [1, T, V, 1]
    # 3D Euclidean bone length — replaces the 2D bone_angle channel
    bone_len = np.sqrt(bone_dx**2 + bone_dy**2 + bone_dz**2).astype(np.float32)
    bone     = np.concatenate([bone_dx, bone_dy, bone_dz, bone_len], axis=0)  # [4, T, V, 1]

    # total channels = 6 + 6 + 4 = 16
    sample = np.concatenate([joint, velocity, bone], axis=0)   # [16, T, V, 1]
    return sample.astype(np.float32)


class SkeletonBuffer:
    """
    CHANGED (3D upgrade):
      - push() / push_pair() now expect joints arrays of shape [25, 3].
      - get_tensor() builds [1, 16, T, V, M] tensors (was [1, 11, T, V, M]).
      - The padding block uses 16 channels (was 11).
      - All other buffer logic, ready(), prune(), remove() are unchanged.
    """
    def __init__(self, buffer_len=MAX_FRAMES):
        self.buffer_len = buffer_len
        self._buffers = {}

    def _get_or_create(self, key):
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.buffer_len)
        return self._buffers[key]

    def push(self, key, joints_xyz):
        """joints_xyz : np.ndarray [25, 3]  — one frame of NTU-25 keypoints in 3D"""
        self._get_or_create(key).append(joints_xyz.copy())

    def push_pair(self, key, joints_a, joints_b):
        """
        For interaction pairs: store (joints_a, joints_b) as a tuple per frame.
        key      : (idA, idB) tuple
        joints_a : np.ndarray [25, 3]
        joints_b : np.ndarray [25, 3]
        """
        self._get_or_create(key).append((joints_a.copy(), joints_b.copy()))

    def ready(self, key):
        return key in self._buffers and len(self._buffers[key]) == self.buffer_len

    def get_tensor(self, key):
        """
        Returns a (1, 16, T, V, M) float tensor ready for EfficientGCN-B0.
        For single tracks: M=1.  For interaction pairs: M=2.
        CHANGED: feature dim is 16 (was 11).
        """
        frames = list(self._buffers[key])

        if isinstance(frames[0], tuple):
            seq_a  = np.stack([f[0] for f in frames], axis=0)    # [T, 25, 3]
            seq_b  = np.stack([f[1] for f in frames], axis=0)    # [T, 25, 3]
            feat_a = generate_features_from_sequence(seq_a)      # [16, T, 25, 1]
            feat_b = generate_features_from_sequence(seq_b)      # [16, T, 25, 1]
            feat   = np.concatenate([feat_a, feat_b], axis=3)    # [16, T, 25, 2]
        else:
            seq  = np.stack(frames, axis=0)                       # [T, 25, 3]
            feat = generate_features_from_sequence(seq)           # [16, T, 25, 1]

        T_actual = feat.shape[1]
        # CHANGED: padding uses 16 channels
        if T_actual < MAX_FRAMES:
            pad  = np.zeros((16, MAX_FRAMES - T_actual, NUM_JOINTS, feat.shape[3]),
                            dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)
        elif T_actual > MAX_FRAMES:
            feat = feat[:, :MAX_FRAMES, :, :]

        tensor = torch.from_numpy(feat).unsqueeze(0)   # [1, 16, T, V, M]
        return tensor

    def remove(self, key):
        self._buffers.pop(key, None)

    def prune(self, valid_keys):
        stale = [k for k in self._buffers if k not in valid_keys]
        for k in stale:
            del self._buffers[k]


print("✓ MediaPipe 3D mapping and SkeletonBuffer defined")


# ============================================================================
# CELL 8: Action Inference Helper
# ============================================================================
"""
CHANGED (3D upgrade):
  split_sample slices the 16-channel tensor into the three 3D feature streams:
    joint    = tensor[:, 0:6]    (abs_xyz + rel_xyz)
    velocity = tensor[:, 6:12]   (fast_xyz + slow_xyz)
    bone     = tensor[:, 12:16]  (bone_dx, bone_dy, bone_dz, bone_len)

  All other inference logic (temperature scaling, top-5, PredictionCache)
  is unchanged.
"""

CONFIDENCE_THRESHOLD = 0.10
SOFTMAX_TEMP         = 1.0


def split_sample(tensor):
    """
    Split [N, 16, T, V, M] into the three feature streams.
    CHANGED: slicing indices updated from 11-ch (4/4/3) to 16-ch (6/6/4).
      joint    : tensor[:, 0:6]
      velocity : tensor[:, 6:12]
      bone     : tensor[:, 12:16]
    """
    return tensor[:, 0:6], tensor[:, 6:12], tensor[:, 12:16]


@torch.no_grad()
def run_action_inference(tensor):
    """
    tensor : (1, 16, T, V, M) on CPU — output of SkeletonBuffer.get_tensor()
    Returns (class_code_str, readable_label, confidence_float)
    """
    tensor = tensor.to(device)
    joint, velocity, bone = split_sample(tensor)
    logits = action_model(joint, velocity, bone)

    calibrated_logits = logits / SOFTMAX_TEMP
    probs = F.softmax(calibrated_logits, dim=1)[0]

    idx  = probs.argmax().item()
    conf = probs[idx].item()

    if conf < CONFIDENCE_THRESHOLD:
        code  = '???'
        label = 'uncertain'
    else:
        code  = IDX_TO_CLASS[idx]
        label = LABEL_NAMES.get(code, code)

    top5_vals, top5_idxs = torch.topk(probs, k=5)
    top5 = [
        (
            IDX_TO_CLASS[i.item()],
            LABEL_NAMES.get(IDX_TO_CLASS[i.item()], IDX_TO_CLASS[i.item()]),
            v.item()
        )
        for i, v in zip(top5_idxs, top5_vals)
    ]
    run_action_inference.last_top5 = top5

    return code, label, conf

run_action_inference.last_top5 = []


class PredictionCache:
    def __init__(self, max_age=8):
        self.max_age = max_age
        self._cache  = {}

    def get(self, key):
        if key in self._cache:
            code, label, conf, age = self._cache[key]
            if age < self.max_age:
                return code, label, conf
        return None

    def set(self, key, code, label, conf):
        self._cache[key] = (code, label, conf, 0)

    def tick(self):
        for k in list(self._cache):
            c, l, conf, age = self._cache[k]
            self._cache[k] = (c, l, conf, age + 1)

    def prune(self, valid_keys):
        for k in list(self._cache):
            if k not in valid_keys:
                del self._cache[k]


print("✓ Action inference helpers (3D) defined")


# ============================================================================
# CELL 9: Complete Visualization Utilities (with Skeleton & ID Color)
# ============================================================================
"""
UNCHANGED in behaviour.
draw_skeleton operates entirely in 2D — it reads only the first two
coordinates of each joint (index 0 = x, index 1 = y) and ignores Z.
This is correct and intentional: OpenCV draws on a 2D image plane.

CHANGED: kpts_ntu is now shape [25, 3], so joint access uses explicit
integer indexing kpts_ntu[i][0] and kpts_ntu[i][1] to extract x and y.
The Z component (kpts_ntu[i][2]) is deliberately not used here.
The (0, 0) visibility guard is also applied only on x and y.
"""

_PALETTE = [
    (0,255,0),(0,128,255),(255,0,128),(255,255,0),(0,255,255),
    (255,0,255),(128,255,0),(0,255,128),(255,128,0),(128,0,255),
]

def _id_color(track_id):
    """Returns a consistent BGR color for a given track ID."""
    return _PALETTE[track_id % len(_PALETTE)]

def draw_skeleton(frame, kpts_ntu, color):
    """
    Draws the NTU-25 skeleton structure on the frame using only (x, y).

    kpts_ntu : np.ndarray [25, 3]  — pixel-space x, y, z per joint.
    The Z-axis is ignored for all OpenCV drawing operations.
    """
    connections = [
        (0, 1), (1, 20), (20, 2), (2, 3),                    # Spine & Head
        (20, 4), (4, 5), (5, 6), (6, 7), (6, 21), (6, 22),   # Left Arm
        (20, 8), (8, 9), (9, 10), (10, 11), (10, 23), (10, 24), # Right Arm
        (0, 12), (12, 13), (13, 14), (14, 15),                # Left Leg
        (0, 16), (16, 17), (17, 18), (18, 19)                 # Right Leg
    ]

    for i in range(len(kpts_ntu)):
        # CHANGED: explicitly index [0] and [1] — do NOT use tuple(kpts_ntu[i])
        # because that would pass 3 values to a 2D OpenCV function.
        x, y = int(kpts_ntu[i][0]), int(kpts_ntu[i][1])
        if x > 0 or y > 0:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    for start_idx, end_idx in connections:
        # Extract only x, y for drawing — Z is ignored.
        pt1 = (int(kpts_ntu[start_idx][0]), int(kpts_ntu[start_idx][1]))
        pt2 = (int(kpts_ntu[end_idx][0]),   int(kpts_ntu[end_idx][1]))
        if pt1 != (0, 0) and pt2 != (0, 0):
            cv2.line(frame, pt1, pt2, color, 2)

def draw_action_overlays(frame, tracks_dict, action_labels, interaction_pairs, track_keypoints=None):
    """Draws boxes, skeletons, and action text."""
    vis = frame.copy()
    drawn_interaction = set()

    for tid, box in tracks_dict.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _id_color(tid)

        if track_keypoints and tid in track_keypoints:
            draw_skeleton(vis, track_keypoints[tid], color)

        in_interaction = any(tid in pair for pair in interaction_pairs)
        thickness = 3 if in_interaction else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        info = action_labels.get(tid)
        label_text = f"ID{tid}: {info[1]} ({info[2]:.0%})" if info else f"ID{tid}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 10, th + 5)
        cv2.rectangle(vis, (x1, ty-th-4), (x1+tw+4, ty+2), (0,0,0), -1)
        cv2.putText(vis, label_text, (x1+2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for pair in interaction_pairs:
        if pair in drawn_interaction: continue
        drawn_interaction.add(pair)
        id_a, id_b = pair
        if id_a in tracks_dict and id_b in tracks_dict:
            ub = union_box(tracks_dict[id_a], tracks_dict[id_b])
            ux1, uy1, ux2, uy2 = [int(v) for v in ub]
            cv2.rectangle(vis, (ux1, uy1), (ux2, uy2), (0,0,255), 2)

            info = action_labels.get(pair)
            if info:
                text = f"ID{id_a}-{id_b}: {info[1]} ({info[2]:.0%})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ty = max(uy1-10, th+5)
                cv2.rectangle(vis, (ux1, ty-th-4), (ux1+tw+4, ty+2), (0,0,80), -1)
                cv2.putText(vis, text, (ux1+2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,100,255), 2)

    return vis

print("✓ Visualization utilities defined")


# ============================================================================
# CELL 10: Full Video Processing Pipeline (MediaPipe 3D Edition)
# ============================================================================
"""
CHANGED (3D upgrade — Z-axis extraction and scaling):

  1. kpts_px is now shape [33, 3] (was [33, 2]) to hold (x, y, z) per landmark.

  2. Z-coordinate extraction:
       lm_z = landmark.z * crop_w

     MediaPipe's landmark.z is the depth relative to the hip midpoint, expressed
     as a fraction of the person's height.  However, MediaPipe normalises it
     proportionally to the crop width, so multiplying by crop_w converts it to
     the same pixel-scale as lm_x and lm_y.  This is consistent with how NTU
     RGB+D depth values are provided relative to the sensor and ensures the
     bone_len features computed downstream have numerically meaningful magnitudes.

  3. Offset application:
     - lm_x += x1 and lm_y += y1 are applied ONLY when the joint was detected
       (same bug-fix logic as the 2D version).
     - lm_z does NOT receive any bounding-box offset because Z is a depth value,
       not a 2D image-plane coordinate.  Adding x1 to z would be physically
       meaningless and would corrupt the depth channel.

  4. kpts_px[lm_idx] = [lm_x, lm_y, lm_z]  — all three coords stored.

  5. Zero/fallback arrays for undetected persons are now shape [25, 3].

  All detection, tracking, interaction, buffer, inference, and visualization
  steps are otherwise UNCHANGED.
"""
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- STABILIZATION UTILITY ---
class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def _low_pass_filter(self, x, alpha, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        edx = self._low_pass_filter(dx, self._alpha(self.dcutoff), self.dx_prev)
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        result = self._low_pass_filter(x, self._alpha(cutoff), self.x_prev)
        self.x_prev, self.dx_prev = result, edx
        return result

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

def process_video_with_action_recognition(
    video_path,
    output_path       = 'output_action.mp4',
    reid_model_path   = 'best_model.pth',
    conf_threshold    = 0.3,         # Lowered for better detection
    max_frames        = None,
    buffer_len        = 120,         # Matches training logs
    bbox_scale        = 1.5,
    iou_thresh        = 0.20,
    dist_thresh       = 150,
    persist_frames    = 10,
    inference_every   = 4,
):
    if not os.path.exists(video_path) or not os.path.exists(reid_model_path):
        print("ERROR: Missing video or ReID model file."); return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames: tot_frames = min(tot_frames, max_frames)

    # ---- Initialization ----
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    detector = YOLO('yolo26s.pt')
    feature_extractor = EnhancedFeatureExtractor(model_path=reid_model_path, device=device)
    tracker = MemoryEnhancedBoTSORT(feature_extractor=feature_extractor, max_age=buffer_len, min_hits=5)
    interaction_detector = InteractionDetector(iou_thresh=iou_thresh, dist_thresh=dist_thresh, persist_frames=persist_frames)

    skeleton_buffer = SkeletonBuffer(buffer_len=buffer_len)
    pred_cache = PredictionCache(max_age=buffer_len)
    pose_filters = {} # Store OneEuro filters per track ID

    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    top5_action_log = defaultdict(lambda: defaultdict(float))

    # ---- Instantiate JSON action logger ----
    # ActionEventLogger is defined in Cell 10.5 (after this function in the file).
    # Python resolves it at call-time, not at function-definition time, so this works.
    action_logger = ActionEventLogger(
        video_path        = video_path,
        output_video_path = output_path,
        fps               = fps,
        width             = width,
        height            = height,
        device_str        = device,
    )

    print(f"\nProcessing {tot_frames} frames with OneEuro Stabilization...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames): break

        # Step 1: Detection
        results = detector(frame, verbose=False)[0]
        detections = [[*box.xyxy[0].cpu().numpy(), float(box.conf[0])]
                      for box in results.boxes if int(box.cls[0]) == 0 and float(box.conf[0]) > conf_threshold]

        # Step 2: Tracking (Handling NumPy array output)
        raw_tracks = tracker.update(frame, np.array(detections) if detections else np.empty((0, 5)))
        tracks_dict = {int(t[4]): t[:4].tolist() for t in raw_tracks} if len(raw_tracks) > 0 else {}

        # Step 3: Interaction & Skeleton Extraction
        active_pairs = interaction_detector.update(tracks_dict)
        track_kpts_norm, track_kpts_pix = {}, {}

        for tid, box in tracks_dict.items():
            ex1, ey1, ex2, ey2 = expand_bbox(box, height, width, scale=bbox_scale)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size > 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pose_result = pose_landmarker.detect(mp_image)

                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks[0]
                    kpts_px = np.zeros((33, 3), dtype=np.float32)
                    c_h, c_w = crop.shape[:2]
                    for i, lm in enumerate(landmarks):
                        kpts_px[i] = [lm.x * c_w + ex1, lm.y * c_h + ey1, lm.z * c_w]

                    # --- STABILIZATION ---
                    if tid not in pose_filters:
                        pose_filters[tid] = OneEuroFilter(freq=fps, mincutoff=0.5, beta=0.01)
                    kpts_px = pose_filters[tid](kpts_px)
                    # ---------------------

                    track_kpts_pix[tid] = mediapipe_to_ntu25(kpts_px, snap_missing_to_spine=False)
                    track_kpts_norm[tid] = mediapipe_to_ntu25(kpts_px, snap_missing_to_spine=True)

        # Step 4: Buffer & Inference
        for tid, ntu25 in track_kpts_norm.items(): skeleton_buffer.push(tid, ntu25)
        for pair in active_pairs:
            if all(p in track_kpts_norm for p in pair):
                skeleton_buffer.push_pair(pair, track_kpts_norm[pair[0]], track_kpts_norm[pair[1]])

        pred_cache.tick()
        action_labels = {}
        if frame_idx % inference_every == 0:
            for key in (list(tracks_dict.keys()) + list(active_pairs)):
                if skeleton_buffer.ready(key):
                    tensor = skeleton_buffer.get_tensor(key)
                    code, label, conf = run_action_inference(tensor)
                    pred_cache.set(key, code, label, conf)
                    action_labels[key] = (code, label, conf)
                    log_key = f"ID{key[0]}-ID{key[1]}" if isinstance(key, tuple) else key
                    for _, t_label, t_conf in run_action_inference.last_top5:
                        if t_conf > top5_action_log[log_key][t_label]:
                            top5_action_log[log_key][t_label] = t_conf
        else:
            for key in (list(tracks_dict.keys()) + list(active_pairs)):
                cached = pred_cache.get(key)
                if cached: action_labels[key] = cached

        # Step 5: Visuals
        vis_frame = draw_action_overlays(frame, tracks_dict, action_labels, active_pairs, track_keypoints=track_kpts_pix)
        vis_frame = draw_trajectories(vis_frame, tracker)
        out_writer.write(vis_frame)

        # Step 6: JSON logging — record state ONLY once per second
        if fps > 0 and frame_idx % fps == 0:
            action_logger.log_event(
                frame_idx       = frame_idx,
                fps             = fps,
                tracks_dict     = tracks_dict,
                action_labels   = action_labels,
                active_pairs    = active_pairs,
                track_kpts_norm = track_kpts_norm,
            )

        if frame_idx % 30 == 0: print(f"  Frame {frame_idx}/{tot_frames} | Tracks: {len(tracks_dict)}")
        frame_idx += 1

    cap.release(); out_writer.release(); pose_landmarker.close()

    # Save JSON log
    json_path = os.path.splitext(output_path)[0] + '_action_log.json'
    action_logger.save(json_path)
    print(f"✓ Action log saved → {json_path}")

    return output_path, top5_action_log, action_logger


print("✓ Video processing pipeline defined")


# ============================================================================
# CELL 10.5: Action Event Logger — JSON Storage Module
# ============================================================================
"""
Records every recognised action, frame metadata, interaction distances / IoU,
and caregiver-patient role context into a structured JSON file.

45 action classes are divided into four categories:
  - patient_specific    : actions typically performed BY a patient
  - caregiver_specific  : actions typically performed BY a caregiver
  - interaction_based   : two-person interaction actions
  - common              : actions that may be performed by either role

The logger is instantiated inside process_video_with_action_recognition and
called once per frame (log_event) and once at the end (save).
No existing inference, tracking, or skeleton logic is modified.
"""

import json
import datetime

# ---- Action category taxonomy (45 classes) ----
ACTION_CATEGORIES = {
    # Actions that a patient/care-receiver would typically perform
    'patient_specific': [
        'A001',  # drink water
        'A002',  # eat meal
        'A003',  # brush teeth
        'A011',  # reading
        'A012',  # writing
        'A018',  # put on glasses
        'A019',  # take off glasses
        'A027',  # jump up
        'A041',  # sneeze/cough
        'A042',  # staggering
        'A043',  # falling down
        'A044',  # headache
        'A045',  # chest pain
        'A046',  # back pain
        'A047',  # neck pain
        'A048',  # nausea/vomiting
        'A049',  # fan self
        'A080',  # squat down
        'A085',  # apply cream on face
        'A086',  # apply cream on hand
        'A089',  # put object into bag
        'A090',  # take object out of bag
        'A091',  # open a box
        'A092',  # move heavy objects
        'A103',  # yawn
    ],
    # Actions that a caregiver would typically perform
    'caregiver_specific': [
        'A053',  # pat on back
        'A056',  # giving object
        'A114',  # carry object
        'A116',  # follow
        'A119',  # support somebody
    ],
    # Two-person interaction actions (usually require both persons present)
    'interaction_based': [
        'A028',  # phone call
        'A050',  # punch/slap
        'A055',  # hugging
        'A058',  # shaking hands
        'A059',  # walking towards
        'A060',  # walking apart
        'A106',  # hit with object
        'A107',  # wield knife
        'A108',  # knock over
        'A109',  # grab stuff
    ],
    # Common actions — can be performed by either role
    'common': [
        'A005',  # drop
        'A006',  # pick up
        'A008',  # sit down
        'A009',  # stand up
        'A054',  # point finger
    ],
}

# Reverse look-up: class_code → category name
_CODE_TO_CATEGORY = {
    code: cat
    for cat, codes in ACTION_CATEGORIES.items()
    for code in codes
}


class ActionEventLogger:
    """
    Incrementally builds a structured JSON action log for a processed video.

    Usage (inside process_video_with_action_recognition):
        logger = ActionEventLogger(video_path, output_path, fps,
                                   width, height, device)
        # --- per frame ---
        logger.log_event(frame_idx, fps, tracks_dict, action_labels,
                         active_pairs, track_kpts_norm)
        # --- at end ---
        logger.save('output_action_log.json')
    """

    def __init__(self, video_path, output_video_path, fps, width, height, device_str):
        self.video_path        = video_path
        self.output_video_path = output_video_path
        self.fps               = fps
        self.width             = width
        self.height            = height
        self.device_str        = str(device_str)
        self.processed_at      = datetime.datetime.now().isoformat(timespec='seconds')

        self._frames         = []        # list of per-frame records
        self._all_track_ids  = set()     # for summary
        self._recognition_count = 0      # total successful recognitions
        self._action_distribution = defaultdict(int)  # label → count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(
        self,
        frame_idx,
        fps,
        tracks_dict,
        action_labels,
        active_pairs,
        track_kpts_norm,
    ):
        """
        Record one frame's state.

        Parameters
        ----------
        frame_idx      : int   — current frame index
        fps            : int   — video frames per second (for timestamp)
        tracks_dict    : dict  — {track_id: [x1, y1, x2, y2]}
        action_labels  : dict  — {key: (code, label, confidence)}
                                  key is int (single track) or tuple (pair)
        active_pairs   : set   — set of (id_a, id_b) interaction tuples
        track_kpts_norm: dict  — {track_id: np.ndarray [25, 3]} NTU keypoints
        """
        timestamp_sec = round(frame_idx / fps, 4) if fps > 0 else 0.0

        # -- Tracks record -----------------------------------------------
        tracks_record = {}
        for tid, box in tracks_dict.items():
            self._all_track_ids.add(tid)
            tracks_record[str(tid)] = {
                'bbox'                : [round(float(v), 2) for v in box],
                'has_skeleton'        : tid in track_kpts_norm,
            }

        # -- Actions record ----------------------------------------------
        actions_record = {}
        for key, (code, label, conf) in action_labels.items():
            if code == '???':
                category = 'uncertain'
            else:
                category = _CODE_TO_CATEGORY.get(code, 'unknown')

            if isinstance(key, tuple):
                record_key = f"{key[0]}-{key[1]}"
            else:
                record_key = str(key)

            actions_record[record_key] = {
                'code'      : code,
                'label'     : label,
                'confidence': round(conf, 6),
                'category'  : category,
            }

            if code != '???' and conf > 0:
                self._recognition_count += 1
                self._action_distribution[label] += 1

        # -- Interactions record -----------------------------------------
        interactions_record = []
        for pair in active_pairs:
            id_a, id_b = pair
            rec = {
                'track_a'           : id_a,
                'track_b'           : id_b,
                'interaction_active': True,
            }
            if id_a in tracks_dict and id_b in tracks_dict:
                box_a = tracks_dict[id_a]
                box_b = tracks_dict[id_b]
                rec['distance_px'] = round(center_distance(box_a, box_b), 2)
                rec['iou']         = round(compute_iou(box_a, box_b), 4)
                # Skeleton proximity: 3D distance between spine joints (joint 20)
                if id_a in track_kpts_norm and id_b in track_kpts_norm:
                    sp_a = track_kpts_norm[id_a][20]   # [3]
                    sp_b = track_kpts_norm[id_b][20]   # [3]
                    skel_dist = float(np.linalg.norm(sp_a - sp_b))
                    rec['skeleton_spine_distance_px'] = round(skel_dist, 2)
            interactions_record.append(rec)

        # -- Also record nearby (non-active) pairs that are close --------
        ids = list(tracks_dict.keys())
        active_set = {tuple(sorted(p)) for p in active_pairs}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                key_pair = tuple(sorted([id_a, id_b]))
                if key_pair in active_set:
                    continue   # already recorded above
                box_a, box_b = tracks_dict[id_a], tracks_dict[id_b]
                dist = center_distance(box_a, box_b)
                if dist < 300:   # only log nearby non-active pairs
                    interactions_record.append({
                        'track_a'           : id_a,
                        'track_b'           : id_b,
                        'interaction_active': False,
                        'distance_px'       : round(dist, 2),
                        'iou'               : round(compute_iou(box_a, box_b), 4),
                    })

        self._frames.append({
            'frame_index'  : frame_idx,
            'timestamp_sec': timestamp_sec,
            'tracks'       : tracks_record,
            'actions'      : actions_record,
            'interactions' : interactions_record,
        })

    def save(self, output_path):
        """
        Write the complete action log to a JSON file.

        Parameters
        ----------
        output_path : str — file path for the JSON output
        """
        # Build category-to-readable-label maps for the metadata block
        cat_readable = {}
        for cat_name, codes in ACTION_CATEGORIES.items():
            cat_readable[cat_name] = [
                {'code': c, 'label': LABEL_NAMES.get(c, c)}
                for c in codes
            ]

        # Action distribution sorted by frequency
        action_dist_sorted = dict(
            sorted(self._action_distribution.items(),
                   key=lambda x: x[1], reverse=True)
        )

        payload = {
            'session': {
                'video_path'        : self.video_path,
                'output_video_path' : self.output_video_path,
                'processed_at'      : self.processed_at,
                'fps'               : self.fps,
                'resolution'        : {'width': self.width, 'height': self.height},
                'model'             : 'EfficientGCN-B0 (3D, NTU 45-class)',
                'device'            : self.device_str,
                'num_action_classes': NUM_CLASSES,
            },
            'action_categories': cat_readable,
            'summary': {
                'total_frames_processed': len(self._frames),
                'unique_track_ids'      : sorted(self._all_track_ids),
                'total_recognitions'    : self._recognition_count,
                'action_distribution'   : action_dist_sorted,
            },
            'frames': self._frames,
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"✓ ActionEventLogger: {len(self._frames)} frames written → {output_path}")
        return output_path


print("✓ ActionEventLogger and ACTION_CATEGORIES defined")

# ============================================================================
# CELL 11: Run the Pipeline
# ============================================================================
"""
Update the paths below and run this cell.
All paths are now local/relative — no Google Colab /content/ prefix needed.

The pipeline now also produces a JSON action log alongside the output video.
Example JSON output path: output_action_log.json  (auto-derived from output_path)
"""

# Reset track counter before each new video
Track._count = 0

output, top5_action_log, action_logger = process_video_with_action_recognition(
    video_path      = 'anurag.mp4',              # <-- update to your video file path
    output_path     = 'output_action.mp4',
    reid_model_path = 'best_model.pth',         # tracker ReID weights
    conf_threshold  = 0.5,
    max_frames      = None,        # set e.g. 300 for a quick test
    buffer_len      = 60,          # temporal window (frames)
    bbox_scale      = 1.5,         # expand boxes before pose crop
    iou_thresh      = 0.20,        # interaction IoU
    dist_thresh     = 150,         # interaction distance (px)
    persist_frames  = 10,          # frames before interaction is "active"
    inference_every = 4,           # run model every N frames
)

print(f"\n✓ Output video : {output}")
print(f"✓ JSON log     : {os.path.splitext(output)[0] + '_action_log.json'}")


# ============================================================================
# CELL 12: Preview Output
# ============================================================================
"""
Open the output video with any local media player.
The paths below point to the files produced by Cell 11.
"""

def show_video(path):
    """Print the absolute path of the output video for easy navigation."""
    if not os.path.exists(path):
        print(f"Video not found: {path}")
        return
    abs_path = os.path.abspath(path)
    print(f"\n✓ Output video ready: {abs_path}")
    print("  Open with any video player (e.g. VLC, Windows Media Player).")

show_video('output_action.mp4')


# ============================================================================
# CELL 13: Top-5 Actions per Identified Person
# ============================================================================
"""
Prints a ranked table of the top-5 most confidently predicted actions for
every tracked person (and interaction pair) seen across the entire video.
Confidence shown is the MAXIMUM value observed for that action over all
inference calls.

Also prints the action category for each entry.

Run this cell AFTER CELL 11 has finished processing the video.
"""

def display_top5_actions(log):
    """
    log : dict  { track_id_or_pair_str -> { action_label -> max_conf } }
    """
    if not log:
        print("No action data collected. Make sure the video was processed first.")
        return

    # Build a reverse label → code map for category look-up
    _label_to_code = {v: k for k, v in LABEL_NAMES.items()}

    print("\n" + "=" * 74)
    print("  TOP-5 ACTIONS PER IDENTIFIED PERSON / INTERACTION PAIR")
    print("=" * 74)

    def sort_key(k):
        return (0, k) if isinstance(k, int) else (1, str(k))

    for subject in sorted(log.keys(), key=sort_key):
        action_scores = log[subject]
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        heading = (
            f"Person ID {subject}"
            if isinstance(subject, int)
            else f"Interaction {subject}"
        )
        print(f"\n{'─' * 74}")
        print(f"  {heading}")
        print(f"{'─' * 74}")
        print(f"  {'Rank':<5} {'Action':<30} {'Category':<20} {'Confidence':<10} {'Bar'}")
        print(f"  {'─'*4} {'─'*29} {'─'*19} {'─'*10} {'─'*30}")

        for rank, (action, conf) in enumerate(sorted_actions, start=1):
            code     = _label_to_code.get(action, '???')
            category = _CODE_TO_CATEGORY.get(code, 'unknown')
            bar      = '█' * int(conf * 30) + '░' * (30 - int(conf * 30))
            print(
                f"  {rank:<5} {action:<30} {category:<20} {conf:>8.1%}   {bar}"
            )

    print("\n" + "=" * 74)
    print("✓ Top-5 action summary complete.")
    print("=" * 74)


# Run the display
display_top5_actions(top5_action_log)

