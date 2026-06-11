"""
╔══════════════════════════════════════════════════════════════════╗
║              PATHWISE AI — Adaptive Learning System             ║
║                  "Learn Smarter. Progress Faster."              ║
║                                                                  ║
║  Complete Google Colab Notebook Implementation                  ║
║  Domain: Data Science & ML Certification Program                ║
║  Paste each ## CELL N ## block into a Colab cell sequentially   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1 — INSTALLATION (run once in Colab)
# !pip install pgmpy networkx plotly ipywidgets pandas numpy scipy matplotlib seaborn -q
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS AND GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# ── FIX 1: get_ipython undefined outside Colab ───────────────────
# Use a safe try/except block instead of calling get_ipython() directly.
# get_ipython() is a Colab/IPython built-in; Pylance doesn't know it.
try:
    # This import only succeeds in IPython / Colab environments
    from IPython import get_ipython as _get_ipython  # type: ignore[import]
    _shell = _get_ipython()
    _in_notebook: bool = _shell is not None
except ImportError:
    _in_notebook = False

import matplotlib
if not _in_notebook:
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# ── FIX 2: plt.Figure is not exported — use matplotlib.figure.Figure ──
from matplotlib.figure import Figure as MplFigure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, pickle, os, logging, copy, warnings
from collections import defaultdict, deque
# ── FIX 3: Optional added to handle None-able parameters ─────────
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output  # type: ignore[import]
    COLAB_MODE = True
except ImportError:
    COLAB_MODE = False

# ── FIX 4: BayesianNetwork renamed to DiscreteBayesianNetwork in pgmpy ≥1.1 ──
# We import the new name with a fallback to the old name for older installs.
PGMPY_AVAILABLE = False
try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as _BayesNetClass  # type: ignore[import]
    except ImportError:
        from pgmpy.models import BayesianNetwork as _BayesNetClass           # type: ignore[import]
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    print("[WARNING] pgmpy not found. Run: !pip install pgmpy")

np.random.seed(42)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('PathWiseAI')

# ── All parameters in one place ──────────────────────────────────
CONFIG: Dict[str, Any] = {
    # Data generation
    'n_students':            200,
    'n_learning_episodes':   20,
    'random_seed':           42,
    'quiz_noise_std':        0.05,
    'init_completion_rate':  0.40,
    # Q-Learning
    'n_train_episodes':      500,
    'alpha_0':               0.30,
    'alpha_decay':           0.001,
    'gamma':                 0.90,
    # Exploration
    'eps_max':               1.00,
    'eps_min':               0.05,
    'eps_decay':             0.005,
    'eps_boost':             0.20,
    'stagnation_threshold':  3,
    # Reward weights  (sum = 1.0)
    'w1_mastery':            0.40,
    'w2_prereq':             0.20,
    'w3_time':               0.20,
    'w4_revisit':            0.10,
    'w5_skip':               0.10,
    # Utility weights (sum = 1.0)
    'wu_mastery':            0.35,
    'wu_time':               0.25,
    'wu_preference':         0.20,
    'wu_goal':               0.20,
    # MDP
    'mastery_threshold':     0.70,
    'mastery_bins':          [0.0, 0.33, 0.70, 1.01],
    'hours_bins':            [0.0, 4.0, 6.0, 9.0],
    # Convergence
    'conv_threshold':        0.001,
    'conv_window':           50,
    # BN integration
    'bn_bonus_weight':       0.15,
    'bn_readiness_min':      0.40,
    # Viz
    'watermark':             'PathWise AI',
    'clr_not_started':       '#FF6B6B',
    'clr_in_progress':       '#FFD93D',
    'clr_mastered':          '#6BCB77',
    'clr_recommended':       '#4D96FF',
    'figure_dpi':            100,
}

assert abs(sum([CONFIG['w1_mastery'], CONFIG['w2_prereq'],
                CONFIG['w3_time'],    CONFIG['w4_revisit'],
                CONFIG['w5_skip']]) - 1.0) < 1e-9, "Reward weights must sum to 1.0"
assert abs(sum([CONFIG['wu_mastery'],    CONFIG['wu_time'],
                CONFIG['wu_preference'], CONFIG['wu_goal']]) - 1.0) < 1e-9, \
    "Utility weights must sum to 1.0"

# ── Fixed 15-concept curriculum ──────────────────────────────────
CONCEPTS: Dict[str, Dict[str, Any]] = {
    'C01': {'name': 'Python Basics',             'difficulty': 1, 'est_hours': 4,  'importance': 0.95, 'ctype': 'programming'},
    'C02': {'name': 'Statistics Fundamentals',    'difficulty': 2, 'est_hours': 6,  'importance': 0.90, 'ctype': 'theory'},
    'C03': {'name': 'Linear Algebra',             'difficulty': 2, 'est_hours': 5,  'importance': 0.85, 'ctype': 'theory'},
    'C04': {'name': 'Data Wrangling (Pandas)',    'difficulty': 2, 'est_hours': 5,  'importance': 0.88, 'ctype': 'programming'},
    'C05': {'name': 'Exploratory Data Analysis',  'difficulty': 3, 'est_hours': 6,  'importance': 0.87, 'ctype': 'practice'},
    'C06': {'name': 'Probability Theory',         'difficulty': 3, 'est_hours': 5,  'importance': 0.83, 'ctype': 'theory'},
    'C07': {'name': 'Machine Learning Basics',    'difficulty': 3, 'est_hours': 8,  'importance': 0.92, 'ctype': 'theory'},
    'C08': {'name': 'Supervised Learning',        'difficulty': 4, 'est_hours': 8,  'importance': 0.93, 'ctype': 'practice'},
    'C09': {'name': 'Unsupervised Learning',      'difficulty': 4, 'est_hours': 7,  'importance': 0.82, 'ctype': 'practice'},
    'C10': {'name': 'Feature Engineering',        'difficulty': 4, 'est_hours': 6,  'importance': 0.85, 'ctype': 'practice'},
    'C11': {'name': 'Model Evaluation',           'difficulty': 4, 'est_hours': 5,  'importance': 0.88, 'ctype': 'practice'},
    'C12': {'name': 'Deep Learning Intro',        'difficulty': 5, 'est_hours': 10, 'importance': 0.80, 'ctype': 'theory'},
    'C13': {'name': 'NLP Fundamentals',           'difficulty': 5, 'est_hours': 8,  'importance': 0.75, 'ctype': 'practice'},
    'C14': {'name': 'MLOps & Deployment',         'difficulty': 5, 'est_hours': 9,  'importance': 0.78, 'ctype': 'programming'},
    'C15': {'name': 'Capstone Project',           'difficulty': 5, 'est_hours': 12, 'importance': 1.00, 'ctype': 'practice'},
}

PREREQUISITES: List[Tuple[str, str]] = [
    ('C01','C02'),('C01','C03'),('C01','C04'),
    ('C02','C05'),('C02','C06'),('C02','C07'),
    ('C03','C07'),('C03','C12'),
    ('C04','C05'),('C04','C10'),
    ('C05','C07'),('C05','C08'),
    ('C06','C07'),('C06','C09'),
    ('C07','C08'),('C07','C09'),
    ('C08','C10'),('C08','C11'),('C08','C12'),
    ('C09','C10'),('C09','C11'),
    ('C10','C11'),('C10','C13'),
    ('C11','C14'),
    ('C12','C13'),
    ('C13','C14'),
    ('C14','C15'),
]

CONCEPT_IDS:    List[str]       = list(CONCEPTS.keys())
CONCEPT_INDEX:  Dict[str, int]  = {cid: i for i, cid in enumerate(CONCEPT_IDS)}
N_CONCEPTS:     int             = len(CONCEPT_IDS)

LEARNING_GOALS       = ['speed', 'depth', 'certification']
LEARNING_PREFERENCES = ['visual', 'reading', 'practice']

PREFERENCE_AFFINITY: Dict[str, Dict[str, float]] = {
    'visual':   {'programming': 0.65, 'theory': 0.55, 'practice': 0.80},
    'reading':  {'programming': 0.60, 'theory': 0.85, 'practice': 0.55},
    'practice': {'programming': 0.85, 'theory': 0.50, 'practice': 0.90},
}

GOAL_IMPORTANCE: Dict[str, Dict[int, float]] = {
    'speed':         {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.45, 5: 0.30},
    'depth':         {1: 0.50, 2: 0.65, 3: 0.80, 4: 0.85, 5: 0.90},
    'certification': {1: 0.70, 2: 0.75, 3: 0.80, 4: 0.85, 5: 0.90},
}

print("✅ PathWise AI — imports and config loaded")
print(f"   {N_CONCEPTS} concepts | {len(PREREQUISITES)} prerequisite edges")


# ═══════════════════════════════════════════════════════════════════
# CELL 3 — SYNTHETIC DATA GENERATION (Module 1: Student Profile Manager)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StudentProfile:
    """Represents a single learner with their current knowledge state."""
    student_id:                 str
    prior_knowledge_vector:     np.ndarray      # 15-dim, initial mastery
    mastery_vector:             np.ndarray      # 15-dim, evolving mastery
    learning_preference:        str             # visual / reading / practice
    available_study_hours:      float           # hours per day
    learning_goal:              str             # speed / depth / certification
    learning_rate:              float           # individual coefficient
    satisfaction_score:         float = 0.5
    completion_rate:            float = 0.40
    episode_history:            List[Dict[str, Any]] = field(default_factory=list)

    def get_goal_encoding(self) -> int:
        return LEARNING_GOALS.index(self.learning_goal)

    def get_mastery_bins(self) -> Tuple[int, ...]:
        """Discretise mastery vector into 3 bins for state encoding."""
        bins = CONFIG['mastery_bins']
        return tuple(int(np.digitize(m, bins) - 1) for m in self.mastery_vector)

    def get_hours_bin(self) -> int:
        return int(np.digitize(self.available_study_hours, CONFIG['hours_bins']) - 1)


class DataGenerator:
    """
    Module 1 — generates 200 synthetic student records with realistic
    learning curves.  Mastery grows via:
        m_new = m_old + lr * (1 - m_old) * affinity
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ── public API ────────────────────────────────────────────────

    def generate_students(self, n: int = CONFIG['n_students']) -> List[StudentProfile]:
        students: List[StudentProfile] = []
        for i in range(n):
            sid  = f"S{i+1:03d}"
            pref = str(self.rng.choice(LEARNING_PREFERENCES))
            goal = str(self.rng.choice(LEARNING_GOALS))
            hours = float(self.rng.uniform(2.0, 8.0))
            lr    = float(self.rng.uniform(0.08, 0.28))

            # Only C01-C03 have non-zero initial mastery
            prior = np.zeros(N_CONCEPTS)
            for j in range(3):
                prior[j] = self.rng.uniform(0.1, 0.65)

            students.append(StudentProfile(
                student_id=sid,
                prior_knowledge_vector=prior.copy(),
                mastery_vector=prior.copy(),
                learning_preference=pref,
                available_study_hours=hours,
                learning_goal=goal,
                learning_rate=lr,
            ))
        return students

    def simulate_learning_episode(self,
                                   student: StudentProfile,
                                   concept_id: str,
                                   kg: 'KnowledgeGraph') -> Dict[str, Any]:
        """
        Simulate one study episode.  Returns updated mastery + metrics.
        """
        idx   = CONCEPT_INDEX[concept_id]
        old_m = student.mastery_vector[idx]

        # Concept-preference affinity
        ctype    = CONCEPTS[concept_id]['ctype']
        affinity = PREFERENCE_AFFINITY[student.learning_preference][ctype]

        # Difficulty penalty: harder concepts → slower gain
        diff_factor = 1.0 - (CONCEPTS[concept_id]['difficulty'] - 1) * 0.10

        # Prerequisite multiplier: gate learning if prereqs not met
        prereqs_met = kg.prerequisites_satisfied(concept_id, student.mastery_vector)
        prereq_mult = 1.0 if prereqs_met else 0.35

        new_m = old_m + student.learning_rate * (1 - old_m) * affinity * diff_factor * prereq_mult
        new_m = float(np.clip(new_m, 0.0, 1.0))
        student.mastery_vector[idx] = new_m

        # Quiz score: mastery + Gaussian noise
        quiz_score = float(np.clip(
            new_m + self.rng.normal(0, CONFIG['quiz_noise_std']), 0, 1))

        # Completion rate increases with momentum
        mastered_count = int(np.sum(student.mastery_vector >= CONFIG['mastery_threshold']))
        student.completion_rate = float(np.clip(
            CONFIG['init_completion_rate'] + 0.03 * mastered_count, 0, 1))

        # Satisfaction: proxy for alignment quality
        satisfaction = float(np.clip(
            0.5 + 0.4 * (new_m - old_m) * 5 + 0.1 * prereq_mult, 0, 1))
        student.satisfaction_score = satisfaction

        record: Dict[str, Any] = {
            'student_id':    student.student_id,
            'concept_id':    concept_id,
            'old_mastery':   old_m,
            'new_mastery':   new_m,
            'delta_mastery': new_m - old_m,
            'quiz_score':    quiz_score,
            'completion_rate': student.completion_rate,
            'satisfaction':  satisfaction,
            'prereqs_met':   prereqs_met,
        }
        student.episode_history.append(record)
        return record

    def build_dataset(self,
                      students: List[StudentProfile],
                      kg: 'KnowledgeGraph') -> pd.DataFrame:
        """Generate the full training dataset across all students × episodes."""
        rows: List[Dict[str, Any]] = []
        rng2 = np.random.default_rng(CONFIG['random_seed'])
        for s in students:
            for ep in range(CONFIG['n_learning_episodes']):
                valid = kg.get_valid_actions(s.mastery_vector)
                if not valid:
                    valid = CONCEPT_IDS[:3]
                # ── FIX 5: rng.choice on list[str] works fine ────
                cid = str(rng2.choice(valid))
                rec = self.simulate_learning_episode(s, cid, kg)
                rec['episode']            = ep
                rec['learning_preference']= s.learning_preference
                rec['learning_goal']      = s.learning_goal
                rec['available_hours']    = s.available_study_hours
                rows.append(rec)
        return pd.DataFrame(rows)


print("✅ DataGenerator and StudentProfile defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 4 — KNOWLEDGE GRAPH ENGINE (Module 3)
# ═══════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Module 3 — NetworkX DiGraph of the 15-concept curriculum.
    Provides prerequisite gating, valid action sets, and layout for viz.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._build()
        # Pre-compute predecessor sets for speed
        self._prereq_cache: Dict[str, List[str]] = {
            cid: list(self.graph.predecessors(cid)) for cid in CONCEPT_IDS
        }

    def _build(self) -> None:
        for cid, attrs in CONCEPTS.items():
            self.graph.add_node(cid, **attrs)
        for src, dst in PREREQUISITES:
            self.graph.add_edge(src, dst)

    def prerequisites_satisfied(self,
                                  concept_id: str,
                                  mastery_vec: np.ndarray,
                                  threshold: Optional[float] = None) -> bool:
        """Return True if all direct prerequisites are above threshold."""
        thr = threshold if threshold is not None else CONFIG['mastery_threshold']
        for p in self._prereq_cache[concept_id]:
            if mastery_vec[CONCEPT_INDEX[p]] < thr:
                return False
        return True

    def get_valid_actions(self, mastery_vec: np.ndarray) -> List[str]:
        """
        Actions are valid when:
          1. All prerequisites ≥ mastery_threshold
          2. Concept is not yet mastered (mastery < 1.0)
        """
        valid = []
        for cid in CONCEPT_IDS:
            idx = CONCEPT_INDEX[cid]
            if (mastery_vec[idx] < 1.0 and
                    self.prerequisites_satisfied(cid, mastery_vec)):
                valid.append(cid)
        return valid if valid else [CONCEPT_IDS[0]]   # fallback: Python Basics

    def get_prerequisite_bonus(self,
                                concept_id: str,
                                mastery_vec: np.ndarray) -> float:
        """Fraction of prerequisites that are mastered."""
        prereqs = self._prereq_cache[concept_id]
        if not prereqs:
            return 1.0
        met = sum(1 for p in prereqs
                  if mastery_vec[CONCEPT_INDEX[p]] >= CONFIG['mastery_threshold'])
        return met / len(prereqs)

    def topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def get_pos_for_viz(self) -> Dict[str, Tuple[float, float]]:
        """Layered layout for visualization."""
        level_map: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        for cid, attrs in CONCEPTS.items():
            level_map[attrs['difficulty']].append(cid)
        pos: Dict[str, Tuple[float, float]] = {}
        for lvl, nodes in level_map.items():
            for j, n in enumerate(nodes):
                x = (j - (len(nodes) - 1) / 2) * 2.5
                pos[n] = (x, float(-lvl * 2))
        return pos

    def node_color(self, mastery: float) -> str:
        if mastery < 0.33:
            return CONFIG['clr_not_started']
        elif mastery < 0.70:
            return CONFIG['clr_in_progress']
        return CONFIG['clr_mastered']


print("✅ KnowledgeGraph defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 5 — BAYESIAN NETWORK ENGINE (Module 4)
# ═══════════════════════════════════════════════════════════════════

class BayesianEngine:
    """
    Module 4 — 5-node Bayesian Network modelling mastery probability.

    Structure:
        PriorKnowledge → LearningAbility → MasteryProbability
        ConceptDifficulty              → MasteryProbability
        LearningAbility                → SuccessProbability
        MasteryProbability             → SuccessProbability

    States:
        PriorKnowledge:    {0=low, 1=medium, 2=high}
        LearningAbility:   {0=low, 1=medium, 2=high}
        ConceptDifficulty: {0=easy(1-2), 1=medium(3), 2=hard(4-5)}
        MasteryProbability:{0=low, 1=medium, 2=high}
        SuccessProbability:{0=fail, 1=pass, 2=distinction}
    """

    def __init__(self) -> None:
        self.model:  Any = None
        self.infer:  Optional[VariableElimination] = None  # type: ignore[type-arg]
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        if PGMPY_AVAILABLE:
            self._build_pgmpy()
        else:
            logger.warning("pgmpy unavailable — using manual CPT fallback")

    # ── pgmpy implementation ───────────────────────────────────────

    def _build_pgmpy(self) -> None:
        # ── FIX 4 (continued): use _BayesNetClass alias ──────────
        # add_cpds and check_model ARE valid methods; Pylance stubs are wrong
        # because the class was renamed.  We type-ignore the attribute calls.
        try:
            model = _BayesNetClass([                             # type: ignore[operator]
                ('PriorKnowledge',    'LearningAbility'),
                ('LearningAbility',   'MasteryProbability'),
                ('ConceptDifficulty', 'MasteryProbability'),
                ('LearningAbility',   'SuccessProbability'),
                ('MasteryProbability','SuccessProbability'),
            ])

            # P(PriorKnowledge) — marginal
            cpd_pk = TabularCPD('PriorKnowledge', 3,
                [[0.35], [0.40], [0.25]])

            # P(LearningAbility | PriorKnowledge)
            cpd_la = TabularCPD('LearningAbility', 3,
                [[0.55, 0.25, 0.10],
                 [0.35, 0.50, 0.35],
                 [0.10, 0.25, 0.55]],
                evidence=['PriorKnowledge'], evidence_card=[3])

            # P(ConceptDifficulty) — marginal
            cpd_cd = TabularCPD('ConceptDifficulty', 3,
                [[0.267], [0.200], [0.533]])

            # P(MasteryProbability | LearningAbility, ConceptDifficulty)
            cpd_mp = TabularCPD('MasteryProbability', 3,
                [[0.360, 0.550, 0.720,  0.150, 0.300, 0.500,  0.050, 0.150, 0.300],
                 [0.450, 0.360, 0.225,  0.450, 0.490, 0.375,  0.295, 0.405, 0.395],
                 [0.190, 0.090, 0.055,  0.400, 0.210, 0.125,  0.655, 0.445, 0.305]],
                evidence=['LearningAbility', 'ConceptDifficulty'],
                evidence_card=[3, 3])

            # P(SuccessProbability | LearningAbility, MasteryProbability)
            cpd_sp = TabularCPD('SuccessProbability', 3,
                [[0.700, 0.395, 0.195,  0.550, 0.245, 0.095,  0.395, 0.145, 0.045],
                 [0.250, 0.480, 0.555,  0.350, 0.520, 0.475,  0.455, 0.480, 0.350],
                 [0.050, 0.125, 0.250,  0.100, 0.235, 0.430,  0.150, 0.375, 0.605]],
                evidence=['LearningAbility', 'MasteryProbability'],
                evidence_card=[3, 3])

            # ── FIX 4a: suppress Pylance attribute errors on pgmpy model ──
            model.add_cpds(cpd_pk, cpd_la, cpd_cd, cpd_mp, cpd_sp)   # type: ignore[attr-defined]
            assert model.check_model(), "BN model check failed"        # type: ignore[attr-defined]
            self.model = model
            self.infer = VariableElimination(model)
            logger.info("Bayesian Network built and validated with pgmpy")
        except Exception as e:
            logger.error(f"pgmpy BN build failed: {e}. Falling back to manual CPT.")
            self.model = None
            self.infer = None

    # ── manual CPT fallback ────────────────────────────────────────

    _CPT_MP = np.array([  # shape (3,3,3): [LA, CD, MP]
        [[0.360, 0.450, 0.190], [0.550, 0.360, 0.090], [0.720, 0.225, 0.055]],
        [[0.150, 0.450, 0.400], [0.300, 0.490, 0.210], [0.500, 0.375, 0.125]],
        [[0.050, 0.295, 0.655], [0.150, 0.405, 0.445], [0.300, 0.395, 0.305]],
    ])

    _CPT_SP = np.array([  # shape (3,3,3): [LA, MP, SP]
        [[0.700, 0.250, 0.050], [0.395, 0.480, 0.125], [0.195, 0.555, 0.250]],
        [[0.550, 0.350, 0.100], [0.245, 0.520, 0.235], [0.095, 0.475, 0.430]],
        [[0.395, 0.455, 0.150], [0.145, 0.480, 0.375], [0.045, 0.350, 0.605]],
    ])

    _CPT_LA = np.array([  # shape (3,3): [PK, LA]
        [0.55, 0.35, 0.10],
        [0.25, 0.50, 0.25],
        [0.10, 0.35, 0.55],
    ])

    def _get_pk_state(self, student: StudentProfile) -> int:
        avg = float(np.mean(student.prior_knowledge_vector[:3]))
        if avg < 0.33:   return 0
        elif avg < 0.67: return 1
        return 2

    def _get_cd_state(self, concept_id: str) -> int:
        d = CONCEPTS[concept_id]['difficulty']
        if d <= 2:   return 0
        elif d == 3: return 1
        return 2

    def _manual_infer(self, pk: int, cd: int) -> Dict[str, np.ndarray]:
        """Compute P(MP) and P(SP) via manual marginalisation."""
        p_la: np.ndarray = self._CPT_LA[pk]
        p_mp = np.zeros(3)
        p_sp = np.zeros(3)
        for la in range(3):
            p_mp += p_la[la] * self._CPT_MP[la, cd]
        for la in range(3):
            for mp in range(3):
                p_sp += p_la[la] * self._CPT_MP[la, cd, mp] * self._CPT_SP[la, mp]
        return {'MasteryProbability': p_mp, 'SuccessProbability': p_sp}

    def infer_mastery(self, student: StudentProfile,
                       concept_id: str) -> Dict[str, np.ndarray]:
        """
        Returns dict with 'MasteryProbability' and 'SuccessProbability',
        each a 3-element probability array.
        """
        cache_key = f"{student.student_id}_{concept_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        pk = self._get_pk_state(student)
        cd = self._get_cd_state(concept_id)

        if self.infer is not None:
            try:
                q = self.infer.query(
                    variables=['MasteryProbability', 'SuccessProbability'],
                    evidence={'PriorKnowledge': pk, 'ConceptDifficulty': cd},
                    show_progress=False,
                )
                # ── FIX 6: DiscreteFactor result accessed via .values (ndarray) ──
                # q['MasteryProbability'] returns a DiscreteFactor; .values is ndarray.
                # Do NOT subscript with [key] — use .values directly.
                result: Dict[str, np.ndarray] = {
                    'MasteryProbability': np.array(q['MasteryProbability'].values),
                    'SuccessProbability': np.array(q['SuccessProbability'].values),
                }
            except Exception:
                result = self._manual_infer(pk, cd)
        else:
            result = self._manual_infer(pk, cd)

        self._cache[cache_key] = result
        return result

    def p_mastery_high(self, student: StudentProfile, concept_id: str) -> float:
        """Scalar: P(MasteryProbability = high)."""
        return float(self.infer_mastery(student, concept_id)['MasteryProbability'][2])

    def p_success_pass(self, student: StudentProfile, concept_id: str) -> float:
        """Scalar: P(SuccessProbability ≥ pass) = P(pass) + P(distinction)."""
        sp = self.infer_mastery(student, concept_id)['SuccessProbability']
        return float(sp[1] + sp[2])

    def is_ready(self, student: StudentProfile, concept_id: str) -> bool:
        """BN gate: recommend only if P(success≥pass) ≥ threshold."""
        return self.p_success_pass(student, concept_id) >= CONFIG['bn_readiness_min']

    def reward_shaping_bonus(self, student: StudentProfile, concept_id: str) -> float:
        """BN-derived bonus added to RL reward."""
        return CONFIG['bn_bonus_weight'] * self.p_mastery_high(student, concept_id)


print("✅ BayesianEngine defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 6 — MDP FORMULATION (Module 5 — partial)
# ═══════════════════════════════════════════════════════════════════

class MDPEnvironment:
    """
    Module 5 — Markov Decision Process for adaptive concept recommendation.

    State:  (mastery_bins[15], hours_bin, goal_encoding)   → hashed int
    Action: concept_id ∈ {C01..C15}
    Reward: 5-component weighted sum + BN bonus
    """

    def __init__(self, kg: KnowledgeGraph, bn: BayesianEngine) -> None:
        self.kg = kg
        self.bn = bn

    def encode_state(self, student: StudentProfile) -> int:
        """Hash the discretised student state to a sparse integer key."""
        bins  = student.get_mastery_bins()
        hbin  = student.get_hours_bin()
        genc  = student.get_goal_encoding()
        state = bins + (hbin, genc)
        return hash(state)

    def compute_reward(self,
                       student: StudentProfile,
                       action_cid: str,
                       new_mastery_vec: np.ndarray) -> float:
        """
        R = w1*ΔMastery + w2*PrereqBonus + w3*TimeEfficiency
              - w4*RevisitPenalty - w5*SkipPenalty + BN_bonus
        """
        idx   = CONCEPT_INDEX[action_cid]
        old_m = student.mastery_vector[idx]
        new_m = new_mastery_vec[idx]
        delta = float(new_m - old_m)

        r1 = float(np.clip(delta * 4.0, 0.0, 1.0))
        r2 = self.kg.get_prerequisite_bonus(action_cid, student.mastery_vector)

        est_h = float(CONCEPTS[action_cid]['est_hours'])
        avail = student.available_study_hours * 2
        r3    = float(np.clip(1.0 - max(0.0, est_h - avail) / (est_h + 1e-9), 0.0, 1.0))

        p4 = 1.0 if old_m >= CONFIG['mastery_threshold'] else 0.0
        prereqs_met = self.kg.prerequisites_satisfied(action_cid, student.mastery_vector)
        p5 = 0.0 if prereqs_met else 1.0

        reward = (CONFIG['w1_mastery']  * r1
                + CONFIG['w2_prereq']   * r2
                + CONFIG['w3_time']     * r3
                - CONFIG['w4_revisit']  * p4
                - CONFIG['w5_skip']     * p5)
        reward += self.bn.reward_shaping_bonus(student, action_cid)
        return float(reward)

    def step(self,
             student: StudentProfile,
             action_cid: str,
             data_gen: DataGenerator) -> Tuple[int, float, StudentProfile]:
        """
        Execute action, return (new_state_hash, reward, updated_student).
        If prerequisites violated → deterministic stay + large negative reward.
        """
        if not self.kg.prerequisites_satisfied(action_cid, student.mastery_vector):
            return self.encode_state(student), -0.30, student

        student_copy = copy.deepcopy(student)
        data_gen.simulate_learning_episode(student_copy, action_cid, self.kg)
        reward    = self.compute_reward(student, action_cid, student_copy.mastery_vector)
        new_state = self.encode_state(student_copy)
        return new_state, reward, student_copy


print("✅ MDPEnvironment defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 7 — Q-LEARNING TRAINING (Module 5 — continued)
# ═══════════════════════════════════════════════════════════════════

# ── Type alias to silence Pylance on defaultdict[int, defaultdict[str, float]] ──
_QTable = Dict[int, Dict[str, float]]

class QLearningAgent:
    """
    Tabular Q-Learning with sparse dict Q-table and decaying α.
    Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') − Q(s,a)]
    """

    def __init__(self) -> None:
        # ── FIX 7: explicit typed defaultdict resolves max() overload error ──
        # Pylance struggles with nested defaultdicts; we cast to _QTable on access.
        self._Q: Any = defaultdict(lambda: defaultdict(float))
        self.episode_rewards:   List[float]         = []
        self.episode_delta_q:   List[float]         = []
        self.converged_at:      Optional[int]       = None
        self._conv_buffer:      deque[float]        = deque(maxlen=CONFIG['conv_window'])

    # ── expose Q as a typed property ─────────────────────────────
    @property
    def Q(self) -> _QTable:
        return self._Q  # type: ignore[return-value]

    # ── helpers ───────────────────────────────────────────────────

    def _alpha(self, t: int) -> float:
        return float(CONFIG['alpha_0'] / (1.0 + CONFIG['alpha_decay'] * t))

    def _max_q(self, state: int, valid_actions: List[str]) -> float:
        if not valid_actions:
            return 0.0
        # ── FIX 7 (continued): iterate list, index defaultdict explicitly ──
        return max(float(self._Q[state][a]) for a in valid_actions)

    def select_action(self,
                       state: int,
                       valid_actions: List[str],
                       epsilon: float,
                       rng: np.random.Generator) -> Tuple[str, bool]:
        """ε-greedy action selection. Returns (action, is_explore)."""
        if not valid_actions:
            return CONCEPT_IDS[0], True
        if rng.random() < epsilon:
            return str(rng.choice(valid_actions)), True
        # ── FIX 7: explicit lambda with float cast removes overload ambiguity ──
        best: str = max(valid_actions, key=lambda a: float(self._Q[state][a]))
        return best, False

    def update(self,
               state: int,
               action: str,
               reward: float,
               next_state: int,
               next_valid: List[str],
               t: int) -> float:
        """Single Bellman update. Returns |ΔQ|."""
        alpha  = self._alpha(t)
        old_q  = float(self._Q[state][action])
        max_nq = self._max_q(next_state, next_valid)
        new_q  = old_q + alpha * (reward + CONFIG['gamma'] * max_nq - old_q)
        self._Q[state][action] = new_q
        return abs(new_q - old_q)

    def check_convergence(self, mean_dq: float, episode: int) -> bool:
        self._conv_buffer.append(mean_dq)
        if (len(self._conv_buffer) == CONFIG['conv_window'] and
                max(self._conv_buffer) < CONFIG['conv_threshold'] and
                self.converged_at is None):
            self.converged_at = episode
            logger.info(f"Q-table converged at episode {episode}")
            return True
        return False

    def get_policy(self, state: int, valid_actions: List[str]) -> Optional[str]:
        if not valid_actions:
            return None
        return max(valid_actions, key=lambda a: float(self._Q[state][a]))


class ExplorationTracker:
    """
    Manages ε-greedy decay and stagnation-based ε-boost.
    ε(t) = ε_max · exp(−decay · t), floored at ε_min.
    """

    def __init__(self) -> None:
        self.epsilon_history:   List[float] = []
        self.explore_flags:     List[bool]  = []
        self.stagnation_events: List[int]   = []
        self._no_gain_streak:   int         = 0
        self._last_avg_mastery: float       = 0.0

    def get_epsilon(self, t: int, stagnated: bool = False) -> float:
        eps = max(CONFIG['eps_min'],
                  CONFIG['eps_max'] * np.exp(-CONFIG['eps_decay'] * t))
        if stagnated:
            eps = min(1.0, eps + CONFIG['eps_boost'])
        return float(eps)

    def update_stagnation(self, current_avg_mastery: float, episode: int) -> bool:
        delta = current_avg_mastery - self._last_avg_mastery
        if delta < 1e-4:
            self._no_gain_streak += 1
        else:
            self._no_gain_streak = 0
        self._last_avg_mastery = current_avg_mastery
        stagnated = self._no_gain_streak >= CONFIG['stagnation_threshold']
        if stagnated:
            self.stagnation_events.append(episode)
            self._no_gain_streak = 0
        return stagnated

    def log(self, epsilon: float, is_explore: bool) -> None:
        self.epsilon_history.append(epsilon)
        self.explore_flags.append(is_explore)

    def explore_ratio_per_window(self, window: int = 20) -> List[float]:
        flags  = self.explore_flags
        ratios: List[float] = []
        for i in range(0, len(flags), window):
            chunk = flags[i:i+window]
            ratios.append(sum(chunk) / len(chunk) if chunk else 0.0)
        return ratios


def run_training(students:  List[StudentProfile],
                 kg:        KnowledgeGraph,
                 bn:        BayesianEngine,
                 mdp:       MDPEnvironment,
                 data_gen:  DataGenerator) -> Tuple[QLearningAgent, ExplorationTracker]:
    """
    Main Q-Learning training loop.
    500 episodes × up to 15 steps each → tracks reward + convergence.
    """
    agent       = QLearningAgent()
    tracker     = ExplorationTracker()
    rng         = np.random.default_rng(CONFIG['random_seed'])
    n_ep        = CONFIG['n_train_episodes']
    global_step = 0

    # ── FIX 8: rng.choice cannot accept List[StudentProfile] ─────
    # Use rng.integers to pick an index, then index into the list.
    n_students = len(students)

    for ep in range(n_ep):
        student = copy.deepcopy(students[int(rng.integers(0, n_students))])
        ep_reward   = 0.0
        dq_list:    List[float] = []
        steps       = 0

        avg_mastery = float(np.mean(student.mastery_vector))
        stagnated   = tracker.update_stagnation(avg_mastery, ep)
        epsilon     = tracker.get_epsilon(ep, stagnated)

        while steps < N_CONCEPTS:
            state      = mdp.encode_state(student)
            valid_acts = kg.get_valid_actions(student.mastery_vector)
            if not valid_acts:
                break

            action, is_explore = agent.select_action(state, valid_acts, epsilon, rng)
            tracker.log(epsilon, is_explore)

            next_state, reward, student = mdp.step(student, action, data_gen)
            next_valid = kg.get_valid_actions(student.mastery_vector)

            dq = agent.update(state, action, reward, next_state, next_valid, global_step)
            dq_list.append(dq)
            ep_reward   += reward
            global_step += 1
            steps       += 1

        agent.episode_rewards.append(ep_reward)
        mean_dq = float(np.mean(dq_list)) if dq_list else 0.0
        agent.episode_delta_q.append(mean_dq)
        agent.check_convergence(mean_dq, ep)

        if ep % 100 == 0:
            logger.info(f"Episode {ep:4d} | ε={epsilon:.3f} | "
                        f"reward={ep_reward:.3f} | ΔQ={mean_dq:.4f}")

    return agent, tracker


print("✅ QLearningAgent, ExplorationTracker, run_training defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 8 — EXPLORATION vs EXPLOITATION
# ═══════════════════════════════════════════════════════════════════

def plot_exploration(tracker:   ExplorationTracker,
                     agent:     QLearningAgent,
                     save_path: Optional[str] = None) -> MplFigure:
    """Plots ε decay, explore/exploit ratio, and stagnation events."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{CONFIG["watermark"]} — Exploration vs Exploitation',
                 fontsize=14, fontweight='bold')

    eps_hist = tracker.epsilon_history
    episodes = list(range(len(agent.episode_rewards)))

    axes[0].plot(eps_hist, color='steelblue', linewidth=1.2, alpha=0.8)
    for ev in tracker.stagnation_events:
        axes[0].axvline(ev * N_CONCEPTS, color='red', alpha=0.5,
                        linestyle='--', linewidth=1)
    axes[0].set_title('ε-Decay with Stagnation Boosts')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('ε (Exploration Rate)')
    axes[0].set_ylim(0, 1.1)

    ratios = tracker.explore_ratio_per_window(window=15)
    axes[1].plot(ratios, color='darkorange', linewidth=1.5, marker='o', markersize=3)
    axes[1].axhline(0.5, linestyle='--', color='gray', alpha=0.6)
    axes[1].set_title('Explore Ratio (window=15 steps)')
    axes[1].set_xlabel('Window Index')
    axes[1].set_ylabel('Fraction Exploring')
    axes[1].set_ylim(0, 1.1)

    mid = len(agent.episode_rewards) // 2
    axes[2].plot(episodes[:mid], agent.episode_rewards[:mid],
                 label='Explore era', color='tomato', alpha=0.7)
    axes[2].plot(episodes[mid:], agent.episode_rewards[mid:],
                 label='Exploit era', color='mediumseagreen', alpha=0.7)
    axes[2].set_title('Cumulative Reward: Explore vs Exploit Era')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Episode Reward')
    axes[2].legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=CONFIG.get('figure_dpi', 100))
    return fig


# ═══════════════════════════════════════════════════════════════════
# CELL 9 — UTILITY THEORY MODULE
# ═══════════════════════════════════════════════════════════════════

class UtilityCalculator:
    """
    Multi-Attribute Utility Function:
        U(a,s) = wu_mastery * U_mastery
               + wu_time    * U_time
               + wu_pref    * U_preference
               + wu_goal    * U_goal_alignment
    """

    def __init__(self, bn: BayesianEngine) -> None:
        self.bn = bn

    def u_mastery(self, student: StudentProfile, concept_id: str) -> float:
        return self.bn.p_mastery_high(student, concept_id)

    def u_time(self, student: StudentProfile, concept_id: str) -> float:
        est   = float(CONCEPTS[concept_id]['est_hours'])
        avail = student.available_study_hours * 2
        return float(np.clip(1.0 - max(0.0, est - avail) / (est + 1e-9), 0.0, 1.0))

    def u_preference(self, student: StudentProfile, concept_id: str) -> float:
        ctype = str(CONCEPTS[concept_id]['ctype'])
        return PREFERENCE_AFFINITY[student.learning_preference].get(ctype, 0.5)

    def u_goal_alignment(self, student: StudentProfile, concept_id: str) -> float:
        diff            = int(CONCEPTS[concept_id]['difficulty'])
        base            = GOAL_IMPORTANCE[student.learning_goal][diff]
        importance_bonus = float(CONCEPTS[concept_id]['importance']) * 0.15
        return float(np.clip(base + importance_bonus, 0.0, 1.0))

    def compute(self, student: StudentProfile, concept_id: str) -> float:
        return (CONFIG['wu_mastery']    * self.u_mastery(student, concept_id)
              + CONFIG['wu_time']       * self.u_time(student, concept_id)
              + CONFIG['wu_preference'] * self.u_preference(student, concept_id)
              + CONFIG['wu_goal']       * self.u_goal_alignment(student, concept_id))

    def rank_actions(self, student: StudentProfile,
                      valid_actions: List[str]) -> List[Tuple[str, float]]:
        scored = [(a, self.compute(student, a)) for a in valid_actions]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def sensitivity_analysis(self,
                              student: StudentProfile,
                              valid_actions: List[str],
                              weight_shifts: Optional[List[Dict[str, Any]]] = None
                              ) -> pd.DataFrame:
        """Show how top recommendation changes when utility weights shift."""
        if weight_shifts is None:
            weight_shifts = [
                {'name': 'Baseline',        'wu_mastery': 0.35, 'wu_time': 0.25, 'wu_preference': 0.20, 'wu_goal': 0.20},
                {'name': 'Mastery Focus',   'wu_mastery': 0.60, 'wu_time': 0.15, 'wu_preference': 0.15, 'wu_goal': 0.10},
                {'name': 'Time Focus',      'wu_mastery': 0.20, 'wu_time': 0.55, 'wu_preference': 0.15, 'wu_goal': 0.10},
                {'name': 'Goal Focus',      'wu_mastery': 0.20, 'wu_time': 0.15, 'wu_preference': 0.15, 'wu_goal': 0.50},
            ]
        rows: List[Dict[str, Any]] = []
        orig = {k: CONFIG[k] for k in ['wu_mastery', 'wu_time', 'wu_preference', 'wu_goal']}
        for scenario in weight_shifts:
            CONFIG.update({k: v for k, v in scenario.items() if k.startswith('wu_')})
            for a in valid_actions:
                rows.append({'Scenario': scenario['name'],
                             'Concept':  a,
                             'Name':     CONCEPTS[a]['name'],
                             'Utility':  round(self.compute(student, a), 4)})
        CONFIG.update(orig)  # restore original weights
        return pd.DataFrame(rows)


print("✅ UtilityCalculator defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 10 — RECOMMENDATION ENGINE (Module 6)
# ═══════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """
    Module 6 — Combines Q-values (primary) + Utility (tiebreaker) + BN gate.
    Returns ranked top-3 concept recommendations with full reasoning.
    """

    def __init__(self,
                 agent:   QLearningAgent,
                 kg:      KnowledgeGraph,
                 bn:      BayesianEngine,
                 utility: UtilityCalculator,
                 mdp:     MDPEnvironment) -> None:
        self.agent   = agent
        self.kg      = kg
        self.bn      = bn
        self.utility = utility
        self.mdp     = mdp

    def recommend(self,
                   student: StudentProfile,
                   top_k:   int = 3) -> List[Dict[str, Any]]:
        """
        Returns list of dicts (max top_k) with recommendation details.
        Ranking: Q-value primary, Utility score tiebreaker.
        Filtered by BN readiness gate.
        """
        state      = self.mdp.encode_state(student)
        valid_acts = self.kg.get_valid_actions(student.mastery_vector)

        ready_acts = [a for a in valid_acts if self.bn.is_ready(student, a)]
        if not ready_acts:
            ready_acts = valid_acts   # fallback: ignore BN gate if all blocked

        scored: List[Dict[str, Any]] = []
        for a in ready_acts:
            q_val     = float(self.agent.Q[state][a])
            util      = self.utility.compute(student, a)
            bn_mp     = self.bn.p_mastery_high(student, a)
            bn_sp     = self.bn.p_success_pass(student, a)
            prereq_ok = self.kg.prerequisites_satisfied(a, student.mastery_vector)
            combined  = q_val + 0.001 * util
            scored.append({
                'concept_id':    a,
                'concept_name':  CONCEPTS[a]['name'],
                'q_value':       round(q_val, 4),
                'utility_score': round(util,  4),
                'bn_mastery_p':  round(bn_mp, 4),
                'bn_success_p':  round(bn_sp, 4),
                'prereq_ok':     prereq_ok,
                'difficulty':    CONCEPTS[a]['difficulty'],
                'est_hours':     CONCEPTS[a]['est_hours'],
                '_combined':     combined,
            })

        scored.sort(key=lambda x: (x['_combined'], x['utility_score']), reverse=True)
        return scored[:top_k]


print("✅ RecommendationEngine defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 11 — VISUALISATION SUITE (10 required plots)
# ═══════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    """Module 7 + 8 — All 10 visualisations + metric computation."""

    def __init__(self,
                 agent:    QLearningAgent,
                 tracker:  ExplorationTracker,
                 students: List[StudentProfile],
                 kg:       KnowledgeGraph,
                 bn:       BayesianEngine,
                 utility:  UtilityCalculator,
                 rec_eng:  RecommendationEngine,
                 dataset:  pd.DataFrame) -> None:
        self.agent    = agent
        self.tracker  = tracker
        self.students = students
        self.kg       = kg
        self.bn       = bn
        self.utility  = utility
        self.rec      = rec_eng
        self.df       = dataset

    # ── VIZ 1: Knowledge Graph ────────────────────────────────────

    def plot_knowledge_graph(self,
                              student:     Optional[StudentProfile] = None,
                              recommended: Optional[str]            = None,
                              save_path:   Optional[str]            = None
                              ) -> MplFigure:
        G   = self.kg.graph
        pos = self.kg.get_pos_for_viz()
        fig, ax = plt.subplots(figsize=(16, 9))

        mastery_vec = student.mastery_vector if student is not None else np.zeros(N_CONCEPTS)
        node_colors: List[str] = []
        for cid in G.nodes():
            if cid == recommended:
                node_colors.append(CONFIG['clr_recommended'])
            else:
                m = mastery_vec[CONCEPT_INDEX[cid]]
                node_colors.append(self.kg.node_color(float(m)))

        nx.draw_networkx(G, pos=pos, ax=ax,
                         node_color=node_colors, node_size=1800,
                         font_size=7, font_weight='bold',
                         arrows=True, arrowsize=15,
                         edge_color='#555555', width=1.5,
                         labels={n: f"{n}\n{CONCEPTS[n]['name'][:12]}"
                                 for n in G.nodes()})

        legend_patches = [
            mpatches.Patch(color=CONFIG['clr_not_started'],  label='Not Started'),
            mpatches.Patch(color=CONFIG['clr_in_progress'],  label='In Progress'),
            mpatches.Patch(color=CONFIG['clr_mastered'],     label='Mastered'),
            mpatches.Patch(color=CONFIG['clr_recommended'],  label='Recommended'),
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
        suffix = f' | Student: {student.student_id}' if student is not None else ''
        ax.set_title(f'{CONFIG["watermark"]} — Knowledge Graph{suffix}',
                     fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        return fig

    # ── VIZ 2: Q-Value Heatmap ────────────────────────────────────

    def plot_q_heatmap(self,
                        top_n_states: int          = 10,
                        save_path:    Optional[str] = None) -> MplFigure:
        state_keys = list(self.agent.Q.keys())[:top_n_states]
        matrix     = np.zeros((len(state_keys), N_CONCEPTS))
        for r, sk in enumerate(state_keys):
            for c, cid in enumerate(CONCEPT_IDS):
                matrix[r, c] = float(self.agent.Q[sk].get(cid, 0.0))

        fig, ax = plt.subplots(figsize=(16, 6))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                       vmin=matrix.min(), vmax=matrix.max())
        ax.set_xticks(range(N_CONCEPTS))
        ax.set_xticklabels([f"{cid}\n{CONCEPTS[cid]['name'][:8]}"
                            for cid in CONCEPT_IDS],
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(state_keys)))
        ax.set_yticklabels([f"S{i+1}" for i in range(len(state_keys))], fontsize=9)
        ax.set_title(f'{CONFIG["watermark"]} — Q-Value Heatmap (top {top_n_states} states)',
                     fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Q-value')
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    # ── VIZ 3: Cumulative Reward Curve ───────────────────────────

    def plot_reward_curve(self,
                           window:    int          = 20,
                           save_path: Optional[str] = None) -> MplFigure:
        rewards  = self.agent.episode_rewards
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().to_numpy()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rewards,  color='lightcoral', alpha=0.4, linewidth=1, label='Raw')
        ax.plot(smoothed, color='crimson',    linewidth=2.0,
                label=f'Smoothed (w={window})')
        if self.agent.converged_at is not None:
            ax.axvline(self.agent.converged_at, color='navy', linestyle='--',
                       label=f'Converged @ ep {self.agent.converged_at}')
        ax.set_title(f'{CONFIG["watermark"]} — Cumulative Reward over Training',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.legend()
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    # ── VIZ 4: ΔQ Convergence ────────────────────────────────────

    def plot_convergence(self, save_path: Optional[str] = None) -> MplFigure:
        dq       = self.agent.episode_delta_q
        smoothed = pd.Series(dq).rolling(20, min_periods=1).mean().to_numpy()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.semilogy(dq,       color='lightblue', alpha=0.5, linewidth=1,   label='Raw ΔQ')
        ax.semilogy(smoothed, color='steelblue', linewidth=2.0,             label='Smoothed ΔQ')
        ax.axhline(CONFIG['conv_threshold'], color='red', linestyle='--',
                   label=f'Threshold={CONFIG["conv_threshold"]}')
        ax.set_title(f'{CONFIG["watermark"]} — Q-Table Convergence (ΔQ per Episode)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean |ΔQ| (log scale)')
        ax.legend()
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    # ── VIZ 5: Mastery Progression ───────────────────────────────

    def plot_mastery_progression(self,
                                  student_id: Optional[str] = None,
                                  save_path:  Optional[str] = None) -> MplFigure:
        sid   = student_id if student_id is not None else self.students[0].student_id
        sdata = self.df[self.df['student_id'] == sid].sort_values('episode')
        if sdata.empty:
            sdata = self.df.groupby('episode')['new_mastery'].mean().reset_index()
            title_suffix = "(Average)"
        else:
            title_suffix = f"({sid})"

        fig, ax = plt.subplots(figsize=(12, 5))
        if 'concept_id' in sdata.columns:
            for cid in sdata['concept_id'].unique():
                cd = sdata[sdata['concept_id'] == cid].sort_values('episode')
                # ── FIX 9: .to_numpy() avoids pandas ExtensionArray plot error ──
                ax.plot(cd['episode'].to_numpy(), cd['new_mastery'].to_numpy(),
                        label=f"{cid}: {CONCEPTS[cid]['name'][:14]}",
                        alpha=0.8, linewidth=1.5)
        else:
            ax.plot(sdata['episode'].to_numpy(),
                    sdata['new_mastery'].to_numpy(), linewidth=2)
        ax.axhline(CONFIG['mastery_threshold'], linestyle='--', color='green',
                   alpha=0.7, label=f'Threshold ({CONFIG["mastery_threshold"]})')
        ax.set_title(f'{CONFIG["watermark"]} — Mastery Progression {title_suffix}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mastery Score')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        return fig

    # ── VIZ 6: Utility Bar Chart ──────────────────────────────────

    def plot_utility_bars(self,
                           student:   StudentProfile,
                           save_path: Optional[str] = None) -> MplFigure:
        valid  = self.kg.get_valid_actions(student.mastery_vector)
        ranked = self.utility.rank_actions(student, valid)

        names  = [CONCEPTS[a]['name'][:18] for a, _ in ranked]
        scores = [float(s) for _, s in ranked]
        colors = ['#4D96FF' if i == 0 else '#A8C8FF' for i in range(len(scores))]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(names[::-1], scores[::-1], color=colors[::-1])
        ax.set_xlabel('Utility Score')
        ax.set_title(f'{CONFIG["watermark"]} — Utility Scores for {student.student_id}',
                     fontsize=13, fontweight='bold')
        ax.axvline(0.5, linestyle='--', color='gray', alpha=0.5)
        for i, (name, score) in enumerate(zip(names[::-1], scores[::-1])):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    # ── VIZ 7: ε Decay ───────────────────────────────────────────

    def plot_epsilon_decay(self, save_path: Optional[str] = None) -> MplFigure:
        return plot_exploration(self.tracker, self.agent, save_path)

    # ── VIZ 8: Mastery Rate Metric ───────────────────────────────

    def plot_mastery_rate(self, save_path: Optional[str] = None) -> MplFigure:
        avg_mr: List[float] = []
        thr = CONFIG['mastery_threshold']
        for ep in sorted(self.df['episode'].unique()):
            ep_data = self.df[self.df['episode'] == ep]
            mr = float((ep_data['new_mastery'] >= thr).mean() * 100)
            avg_mr.append(mr)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(avg_mr, color='mediumseagreen', linewidth=2.5)
        ax.fill_between(range(len(avg_mr)), avg_mr, alpha=0.2, color='mediumseagreen')
        ax.set_title(f'{CONFIG["watermark"]} — Mastery Rate (MR) over Episodes',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mastery Rate (%)')
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    # ── VIZ 9: Satisfaction Trend ─────────────────────────────────

    def plot_satisfaction_trend(self, save_path: Optional[str] = None) -> MplFigure:
        avg_sat = self.df.groupby('episode')['satisfaction'].mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        # ── FIX 9 (continued): .to_numpy() for pandas index/values ──
        ax.plot(avg_sat.index.to_numpy(), avg_sat.to_numpy(),
                color='darkorchid', linewidth=2.5)
        ax.fill_between(avg_sat.index.to_numpy(), avg_sat.to_numpy(),
                        alpha=0.15, color='darkorchid')
        ax.set_title(f'{CONFIG["watermark"]} — Student Satisfaction Score Trend',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Satisfaction Score')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    # ── VIZ 10: Learning Efficiency Comparison ───────────────────

    def plot_learning_efficiency(self,
                                  baseline_le: float,
                                  pathwise_le: float,
                                  save_path:   Optional[str] = None) -> MplFigure:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(['Random Baseline', 'PathWise AI'],
                      [baseline_le, pathwise_le],
                      color=['#FF6B6B', '#6BCB77'],
                      edgecolor='black', width=0.5)
        for bar, val in zip(bars, [baseline_le, pathwise_le]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{val:.2f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        improvement = (pathwise_le - baseline_le) / baseline_le * 100
        ax.set_title(f'{CONFIG["watermark"]} — Learning Efficiency\n'
                     f'Improvement: +{improvement:.1f}% over baseline',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Learning Efficiency (concepts/hour × 100)')
        ax.set_ylim(0, max(baseline_le, pathwise_le) * 1.3)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
        return fig

    def render_all(self,
                    student:    Optional[StudentProfile] = None,
                    output_dir: str = '.') -> None:
        """Render and save all 10 visualisations."""
        os.makedirs(output_dir, exist_ok=True)
        s = student if student is not None else self.students[0]
        self.plot_knowledge_graph(s,             save_path=f'{output_dir}/01_knowledge_graph.png')
        self.plot_q_heatmap(                      save_path=f'{output_dir}/02_q_heatmap.png')
        self.plot_reward_curve(                   save_path=f'{output_dir}/03_reward_curve.png')
        self.plot_convergence(                    save_path=f'{output_dir}/04_convergence.png')
        self.plot_mastery_progression(s.student_id, save_path=f'{output_dir}/05_mastery_prog.png')
        self.plot_utility_bars(s,                 save_path=f'{output_dir}/06_utility_bars.png')
        self.plot_epsilon_decay(                  save_path=f'{output_dir}/07_epsilon_decay.png')
        self.plot_mastery_rate(                   save_path=f'{output_dir}/08_mastery_rate.png')
        self.plot_satisfaction_trend(             save_path=f'{output_dir}/09_satisfaction.png')

        _ev      = EvaluationEngine(self.students, self.df, self.utility, self.kg)
        _metrics = _ev.full_report()
        self.plot_learning_efficiency(
            _metrics['baseline_le'], _metrics['pathwise_le'],
            save_path=f'{output_dir}/10_learning_efficiency.png',
        )
        print(f"✅ All 10 visualisations saved to {output_dir}/")


print("✅ AnalyticsEngine defined (10 visualisations)")


# ═══════════════════════════════════════════════════════════════════
# CELL 12 — EVALUATION METRICS (Module 7)
# ═══════════════════════════════════════════════════════════════════

class EvaluationEngine:
    """
    Computes the three core PathWise AI metrics:
        LE  = (concepts_mastered / total_study_hours) × 100
        MR  = (concepts with mastery ≥ 0.7 / 15)     × 100
        SS  = Σ(utility_alignment_score) / n_recs
    """

    def __init__(self,
                 students: List[StudentProfile],
                 dataset:  pd.DataFrame,
                 utility:  UtilityCalculator,
                 kg:       KnowledgeGraph) -> None:
        self.students = students
        self.df       = dataset
        self.utility  = utility
        self.kg       = kg

    def learning_efficiency(self,
                             recs:        List[Dict[str, Any]],
                             total_hours: float) -> float:
        mastered = sum(1 for r in recs
                       if float(r.get('new_mastery', 0)) >= CONFIG['mastery_threshold'])
        if total_hours < 1e-6:
            return 0.0
        return (mastered / total_hours) * 100

    def mastery_rate(self, student: StudentProfile) -> float:
        mastered = int(np.sum(student.mastery_vector >= CONFIG['mastery_threshold']))
        return (mastered / N_CONCEPTS) * 100

    def satisfaction_score(self,
                            recommendations: List[Dict[str, Any]],
                            student: StudentProfile) -> float:
        if not recommendations:
            return 0.0
        scores = [float(r.get('utility_score', 0.0)) for r in recommendations]
        return float(np.mean(scores))

    def baseline_le(self,
                     n_episodes: int = CONFIG['n_learning_episodes']) -> float:
        """Random policy LE."""
        rng_b          = np.random.default_rng(CONFIG['random_seed'] + 99)
        total_mastered = 0
        total_hours    = 0.0
        for s in self.students[:50]:
            m = s.prior_knowledge_vector.copy()
            for _ in range(n_episodes):
                cid        = str(rng_b.choice(CONCEPT_IDS))
                idx        = CONCEPT_INDEX[cid]
                m[idx]     = min(1.0, m[idx] + 0.05)
                total_hours += float(CONCEPTS[cid]['est_hours'])
            total_mastered += int(np.sum(m >= CONFIG['mastery_threshold']))
        return (total_mastered / total_hours) * 100 if total_hours > 0 else 0.0

    def pathwise_le(self) -> float:
        total_m = 0
        total_h = 0.0
        for s in self.students[:50]:
            total_m += int(np.sum(s.mastery_vector >= CONFIG['mastery_threshold']))
            total_h += s.available_study_hours * CONFIG['n_learning_episodes']
        return (total_m / total_h) * 100 if total_h > 0 else 0.0

    def full_report(self) -> Dict[str, Any]:
        b_le       = self.baseline_le()
        p_le       = self.pathwise_le()
        avg_mr     = float(np.mean([self.mastery_rate(s) for s in self.students]))
        avg_sat    = float(self.df['satisfaction'].mean())
        improvement = (p_le - b_le) / b_le * 100 if b_le > 0 else 0.0
        return {
            'baseline_le':          round(b_le, 3),
            'pathwise_le':          round(p_le, 3),
            'le_improvement_pct':   round(improvement, 1),
            'avg_mastery_rate':     round(avg_mr, 2),
            'avg_satisfaction':     round(avg_sat, 4),
        }


print("✅ EvaluationEngine defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 13 — UNIT TESTS
# ═══════════════════════════════════════════════════════════════════

def run_unit_tests(kg:       KnowledgeGraph,
                   bn:       BayesianEngine,
                   agent:    QLearningAgent,
                   utility:  UtilityCalculator,
                   students: List[StudentProfile]) -> bool:
    """Run all unit / functional / edge-case tests."""
    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = "") -> None:
        nonlocal passed, failed
        if not condition:
            failed += 1
            print(f"  ❌ [FAIL] {name}: {msg}")
        else:
            passed += 1
            print(f"  ✅ [PASS] {name}")

    print("\n── KnowledgeGraph Tests ──")
    check("KG has 15 nodes",       len(kg.graph.nodes()) == 15)
    check("KG has correct edges",  len(kg.graph.edges()) == len(PREREQUISITES))
    check("C15 has no successors", len(list(kg.graph.successors('C15'))) == 0)
    check("C01 satisfiable zero mastery",
          kg.prerequisites_satisfied('C01', np.zeros(N_CONCEPTS)))
    check("C15 blocked zero mastery",
          not kg.prerequisites_satisfied('C15', np.zeros(N_CONCEPTS)))

    print("\n── BayesianNetwork Tests ──")
    s0   = students[0]
    prob = bn.p_mastery_high(s0, 'C01')
    check("BN mastery prob ∈ [0,1]", 0.0 <= prob <= 1.0)
    sp   = bn.p_success_pass(s0, 'C07')
    check("BN success prob ∈ [0,1]", 0.0 <= sp <= 1.0)
    result = bn.infer_mastery(s0, 'C01')
    mp_sum = float(np.sum(result['MasteryProbability']))
    check("BN MasteryProbability sums to 1",  abs(mp_sum - 1.0) < 0.01, f"got {mp_sum}")
    sp_sum = float(np.sum(result['SuccessProbability']))
    check("BN SuccessProbability sums to 1",  abs(sp_sum - 1.0) < 0.01, f"got {sp_sum}")

    print("\n── Utility Tests ──")
    u_val = utility.compute(s0, 'C01')
    check("Utility ∈ [0,1]", 0.0 <= u_val <= 1.0)
    w_sum = (CONFIG['wu_mastery'] + CONFIG['wu_time'] +
             CONFIG['wu_preference'] + CONFIG['wu_goal'])
    check("Utility weights sum to 1.0", abs(w_sum - 1.0) < 1e-9)
    ranked = utility.rank_actions(s0, ['C01', 'C02', 'C04'])
    check("Utility ranking non-empty", len(ranked) > 0)

    print("\n── Q-Learning Tests ──")
    rng_t = np.random.default_rng(0)
    st    = kg.get_valid_actions(s0.mastery_vector)
    _, flag_explore  = agent.select_action(0, st, epsilon=1.0, rng=rng_t)
    check("Epsilon=1 always explores",  flag_explore)
    _, flag_exploit = agent.select_action(0, st, epsilon=0.0, rng=rng_t)
    check("Epsilon=0 always exploits",  not flag_exploit)
    dq = agent.update(0, 'C01', 0.5, 1, ['C01'], t=1)
    check("Q-update returns non-negative |ΔQ|", dq >= 0)

    print("\n── Edge Cases ──")
    full_m    = np.ones(N_CONCEPTS)
    full_m[0] = 0.0
    valid_all_mastered = kg.get_valid_actions(full_m)
    check("C15 not in valid when prereqs unmet", 'C15' not in valid_all_mastered)
    check("Fallback returns at least one action", len(kg.get_valid_actions(np.ones(N_CONCEPTS))) >= 1)

    print(f"\n── Results: {passed} passed / {failed} failed ──")
    return failed == 0


print("✅ Unit tests defined")


# ═══════════════════════════════════════════════════════════════════
# CELL 14 — SAVE ARTEFACTS
# ═══════════════════════════════════════════════════════════════════

def save_artefacts(agent:      QLearningAgent,
                   tracker:    ExplorationTracker,
                   students:   List[StudentProfile],
                   metrics:    Dict[str, Any],
                   output_dir: str = '.') -> None:
    """Pickle Q-table and save JSON artefacts for Streamlit app."""
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/q_table.pkl', 'wb') as f:
        pickle.dump(dict(agent.Q), f)

    student_data = [
        {
            'student_id':          s.student_id,
            'mastery_vector':      s.mastery_vector.tolist(),
            'prior_knowledge':     s.prior_knowledge_vector.tolist(),
            'learning_preference': s.learning_preference,
            'learning_goal':       s.learning_goal,
            'available_hours':     s.available_study_hours,
            'satisfaction':        s.satisfaction_score,
            'completion_rate':     s.completion_rate,
        }
        for s in students
    ]
    with open(f'{output_dir}/students.json', 'w') as f:
        json.dump(student_data, f, indent=2)

    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    training_history: Dict[str, Any] = {
        'episode_rewards':   agent.episode_rewards,
        'episode_delta_q':   agent.episode_delta_q,
        'converged_at':      agent.converged_at,
        'epsilon_history':   tracker.epsilon_history[:1000],
        'stagnation_events': tracker.stagnation_events,
    }
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"✅ Artefacts saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════
# MAIN — Orchestration (Run All in Colab)
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__' or True:
    print("\n" + "═"*65)
    print("  PathWise AI — Training Pipeline Starting")
    print("═"*65)

    kg       = KnowledgeGraph()
    print(f"Step 1 ✅ KG: {len(kg.graph.nodes())} nodes, {len(kg.graph.edges())} edges")

    bn       = BayesianEngine()
    print(f"Step 2 ✅ BN: pgmpy={'yes' if bn.model else 'fallback'}")

    rng      = np.random.default_rng(CONFIG['random_seed'])
    data_gen = DataGenerator(rng)
    students = data_gen.generate_students(CONFIG['n_students'])
    dataset  = data_gen.build_dataset(students, kg)
    print(f"Step 3 ✅ Data: {len(students)} students × "
          f"{CONFIG['n_learning_episodes']} episodes = {len(dataset)} rows")

    mdp     = MDPEnvironment(kg, bn)
    utility = UtilityCalculator(bn)
    print("Step 4 ✅ MDP + Utility ready")

    print(f"\nStep 5 — Q-Learning: {CONFIG['n_train_episodes']} episodes...")
    agent, tracker = run_training(students, kg, bn, mdp, data_gen)
    print(f"Step 5 ✅ Training complete | Q-table entries: {len(agent.Q)}")

    rec_eng  = RecommendationEngine(agent, kg, bn, utility, mdp)
    print("Step 6 ✅ RecommendationEngine ready")

    analytics = AnalyticsEngine(agent, tracker, students, kg, bn, utility, rec_eng, dataset)
    evaluator = EvaluationEngine(students, dataset, utility, kg)
    metrics   = evaluator.full_report()
    print(f"Step 7 ✅ Metrics: {metrics}")

    print("\nStep 8 — Unit Tests:")
    run_unit_tests(kg, bn, agent, utility, students)

    analytics.render_all(students[0], output_dir='./pathwise_output')
    save_artefacts(agent, tracker, students, metrics, output_dir='./pathwise_output')

    s_demo = students[0]
    recs   = rec_eng.recommend(s_demo, top_k=3)
    print(f"\n📌 Demo recommendation for {s_demo.student_id}:")
    for i, r in enumerate(recs):
        print(f"   {i+1}. {r['concept_id']} — {r['concept_name']}"
              f"  Q={r['q_value']:.4f}  U={r['utility_score']:.4f}"
              f"  BN_MP={r['bn_mastery_p']:.4f}")

    if COLAB_MODE:
        student_ids = [s.student_id for s in students[:20]]
        dd_student  = widgets.Dropdown(            # type: ignore[union-attr]
            options=student_ids,
            description='Student:',
            value=student_ids[0],
        )
        out_widget = widgets.Output()              # type: ignore[union-attr]

        def on_change(change: Dict[str, Any]) -> None:
            if change['type'] == 'change' and change['name'] == 'value':
                sid  = str(change['new'])
                s    = next(x for x in students if x.student_id == sid)
                recs_w = rec_eng.recommend(s, top_k=3)
                with out_widget:
                    clear_output(wait=True)        # type: ignore[union-attr]
                    print(f"\n📌 Recommendations for {sid} "
                          f"(goal={s.learning_goal}, pref={s.learning_preference}):")
                    for i, r in enumerate(recs_w):
                        print(f"  {i+1}. {r['concept_id']} — {r['concept_name']}")
                        print(f"      Q={r['q_value']:.4f} | "
                              f"Utility={r['utility_score']:.4f} | "
                              f"BN_mastery={r['bn_mastery_p']:.4f} | "
                              f"BN_success={r['bn_success_p']:.4f}")
                    fig = analytics.plot_knowledge_graph(s)
                    plt.show()
                    plt.close()

        dd_student.observe(on_change)
        on_change({'type': 'change', 'name': 'value', 'new': student_ids[0]})
        display(                                   # type: ignore[union-attr]
            widgets.VBox([dd_student, out_widget]) # type: ignore[union-attr]
        )

    print("\n" + "═"*65)
    print("  PathWise AI — Pipeline Complete ✅")
    print("═"*65)