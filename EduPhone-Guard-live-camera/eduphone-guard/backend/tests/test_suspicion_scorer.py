from app.schemas.pose import PoseSignals
from app.vision.pose import SuspicionScorer


def _scorer(**overrides):
    defaults = dict(
        weight_phone_visual=0.55,
        weight_hand_proximity=0.20,
        weight_head_down=0.10,
        weight_hands_hidden=0.10,
        weight_persistence=0.10,
        weight_writing_reduction=0.25,
    )
    defaults.update(overrides)
    return SuspicionScorer(**defaults)


def test_phone_confidence_alone_contributes_base_score():
    scorer = _scorer()
    result = scorer.score(phone_confidence=0.9, pose_signals=PoseSignals())
    assert result.suspicion_score == 0.55 * 0.9
    assert any("detecção visual de celular" in r for r in result.reasons)


def test_hands_near_phone_increases_score():
    scorer = _scorer()
    base = scorer.score(phone_confidence=0.9, pose_signals=PoseSignals()).suspicion_score
    boosted = scorer.score(
        phone_confidence=0.9, pose_signals=PoseSignals(hands_near_phone=True)
    ).suspicion_score
    assert boosted > base
    assert boosted - base == 0.20


def test_writing_pose_reduces_score():
    scorer = _scorer()
    base = scorer.score(
        phone_confidence=0.9, pose_signals=PoseSignals(hands_near_phone=True)
    ).suspicion_score
    reduced = scorer.score(
        phone_confidence=0.9,
        pose_signals=PoseSignals(hands_near_phone=True, writing_or_reading_pose=True),
    ).suspicion_score
    assert reduced < base


def test_score_is_clamped_between_zero_and_one():
    scorer = _scorer(weight_phone_visual=2.0, weight_hand_proximity=2.0, weight_head_down=2.0, weight_hands_hidden=2.0)
    result = scorer.score(
        phone_confidence=1.0,
        pose_signals=PoseSignals(hands_near_phone=True, head_down=True, hands_hidden_below_hip=True),
    )
    assert result.suspicion_score == 1.0

    scorer_negative = _scorer(weight_writing_reduction=5.0)
    result_negative = scorer_negative.score(
        phone_confidence=0.1, pose_signals=PoseSignals(writing_or_reading_pose=True)
    )
    assert result_negative.suspicion_score == 0.0


def test_reasons_are_populated_and_explainable():
    scorer = _scorer()
    result = scorer.score(
        phone_confidence=0.8,
        pose_signals=PoseSignals(hands_near_phone=True, head_down=True),
    )
    assert len(result.reasons) >= 3
    assert result.phone_confidence == 0.8
    assert result.pose_signals.hands_near_phone is True


def test_never_asserts_phone_presence_beyond_given_confidence():
    """
    O scorer nunca deve "inventar" confiança de celular a partir de sinais
    de pose — phone_confidence no resultado deve ser exatamente o valor
    de entrada, não inflado pelos sinais de postura.
    """
    scorer = _scorer()
    result = scorer.score(
        phone_confidence=0.71,
        pose_signals=PoseSignals(hands_near_phone=True, head_down=True, hands_hidden_below_hip=True),
    )
    assert result.phone_confidence == 0.71
