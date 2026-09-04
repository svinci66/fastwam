from experiments.robotwin.fastwam_policy.deploy_policy import (
    _supported_keyword_arguments,
)


class _Model:
    def infer_action(self, *, prompt, return_video_expert_feature=False):
        del prompt, return_video_expert_feature

    def infer_joint(self, *, prompt, num_video_frames):
        del prompt, num_video_frames

    def flexible(self, **kwargs):
        del kwargs


def test_filters_action_only_video_feature_from_joint_inference():
    model = _Model()
    values = {
        "prompt": "task",
        "num_video_frames": 7,
        "return_video_expert_feature": True,
    }

    action = _supported_keyword_arguments(model.infer_action, values)
    joint = _supported_keyword_arguments(model.infer_joint, values)

    assert action == {"prompt": "task", "return_video_expert_feature": True}
    assert joint == {"prompt": "task", "num_video_frames": 7}


def test_preserves_keywords_for_flexible_entrypoint():
    values = {"prompt": "task", "future_option": 1}
    assert _supported_keyword_arguments(_Model().flexible, values) == values
