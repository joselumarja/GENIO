from genio import Candidate, Evaluator, LocalBackend, Result, ResultStatus, Runner


class SumRunner(Runner):
    def run(self, candidate: Candidate) -> Result:
        return Result.success(candidate.id, {"score": sum(candidate.parameters.values())})


def test_evaluator_returns_results_for_candidates():
    evaluator = Evaluator(LocalBackend(SumRunner()))

    results = evaluator.evaluate([
        Candidate("candidate_001", {"a": 1, "b": 2}),
        Candidate("candidate_002", {"a": 3, "b": 4}),
    ])

    assert [result.candidate_id for result in results] == ["candidate_001", "candidate_002"]
    assert [result.status for result in results] == [ResultStatus.SUCCESS, ResultStatus.SUCCESS]
    assert [result.metrics["score"] for result in results] == [3, 7]
