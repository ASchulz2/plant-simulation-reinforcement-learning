from __future__ import annotations

from dataclasses import dataclass, field

from learning.episode_sampler import EpisodeSampler
from learning.ppo_trainer import PPOTrainer
from learning.ppo_visualizer import PPOVisualizer


@dataclass(frozen=True)
class CurriculumStage:
    """
    One stage of the manufacturing curriculum.

    Storage capacities are changed before the first episode of
    the stage. The actor-critic and PPO optimizer are retained.
    """

    storage_a_capacity: int
    storage_b_capacity: int

    episodes: int = 1

    def __post_init__(self) -> None:

        if self.storage_a_capacity <= 0:
            raise ValueError(
                "Storage A capacity must be positive."
            )

        if self.storage_b_capacity <= 0:
            raise ValueError(
                "Storage B capacity must be positive."
            )

        if self.episodes <= 0:
            raise ValueError(
                "Number of stage episodes must be positive."
            )


@dataclass(frozen=True)
class CurriculumSettings:
    """Configuration of curriculum learning."""

    stages: tuple[CurriculumStage, ...] = (
        CurriculumStage(
            storage_a_capacity=2,
            storage_b_capacity=2,
            episodes=2,
        ),
        CurriculumStage(
            storage_a_capacity=4,
            storage_b_capacity=4,
            episodes=2,
        ),
        CurriculumStage(
            storage_a_capacity=5,
            storage_b_capacity=5,
            episodes=2,
        ),
        CurriculumStage(
            storage_a_capacity=8,
            storage_b_capacity=8,
            episodes=2,
        ),
        CurriculumStage(
            storage_a_capacity=10,
            storage_b_capacity=10,
            episodes=2,
        ),
    )

    # Fixed reward batch size.
    #
    # The environment capacity should not silently redefine
    # the reward function during curriculum learning.
    batch_size: int = 10


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one manufacturing episode."""

    curriculum_stage: int
    episode_index: int

    storage_a_capacity: int
    storage_b_capacity: int

    succeeded: bool
    failed: bool

    throughput: int
    simulation_time: float

    rollouts: int
    decisions: int


@dataclass
class CurriculumHistory:
    """Results collected from curriculum learning."""

    episodes: list[EpisodeResult] = field(
        default_factory=list
    )

    def append(
        self,
        result: EpisodeResult,
    ) -> None:
        self.episodes.append(result)


class CurriculumLearner:
    """
    Train one persistent actor-critic across a sequence of
    manufacturing environments with increasing storage capacity.

    The policy and PPO optimizer are not reset between stages.
    Plant Simulation is reset between episodes.
    """

    def __init__(
        self,
        *,
        plant,
        environment,
        episode_sampler: EpisodeSampler,
        ppo_trainer: PPOTrainer,
        visualizer: PPOVisualizer | None = None,
        settings: CurriculumSettings | None = None,
    ) -> None:

        self.plant = plant
        self.environment = environment
        self.episode_sampler = episode_sampler
        self.ppo_trainer = ppo_trainer

        self.visualizer = (
            visualizer
            if visualizer is not None
            else PPOVisualizer()
        )

        self.settings = (
            settings
            if settings is not None
            else CurriculumSettings()
        )

        if not self.settings.stages:
            raise ValueError(
                "Curriculum must contain at least one stage."
            )

        self.history = CurriculumHistory()

        self._total_episodes = 0
        self._total_decisions = 0

    # ==================================================================
    # Public training interface
    # ==================================================================

    def run(self) -> CurriculumHistory:
        """
        Run the complete curriculum.

        Returns
        -------
        CurriculumHistory
            Episode-level results for all curriculum stages.
        """

        for stage_index, stage in enumerate(
            self.settings.stages,
            start=1,
        ):

            self._configure_stage(
                stage
            )

            print()
            print("=" * 80)
            print(
                f"CURRICULUM STAGE {stage_index}/"
                f"{len(self.settings.stages)}"
            )
            print("=" * 80)

            print(
                "Storage A capacity:",
                stage.storage_a_capacity,
            )

            print(
                "Storage B capacity:",
                stage.storage_b_capacity,
            )

            print(
                "Episodes:",
                stage.episodes,
            )

            for episode_index in range(
                1,
                stage.episodes + 1,
            ):

                result = self._run_episode(
                    stage_index=stage_index,
                    stage=stage,
                    episode_index=episode_index,
                )

                self.history.append(
                    result
                )

                self._print_episode_result(
                    result
                )

        return self.history

    # ==================================================================
    # Curriculum stage
    # ==================================================================

    def _configure_stage(
        self,
        stage: CurriculumStage,
    ) -> None:
        """
        Configure Plant Simulation for one curriculum stage.

        The policy and PPO optimizer remain untouched.
        """

        self.plant.set_model_value(
            relative_path="StorageA.Capacity",
            value=stage.storage_a_capacity,
        )

        self.plant.set_model_value(
            relative_path="StorageB.Capacity",
            value=stage.storage_b_capacity,
        )

    # ==================================================================
    # Episode
    # ==================================================================

    def _run_episode(
        self,
        *,
        stage_index: int,
        stage: CurriculumStage,
        episode_index: int,
    ) -> EpisodeResult:

        self._total_episodes += 1

        print()
        print(
            "=" * 80
        )

        print(
            f"EPISODE {self._total_episodes} "
            f"(Stage {stage_index}, "
            f"Stage Episode {episode_index})"
        )

        print(
            "=" * 80
        )

        # --------------------------------------------------------------
        # Reset Plant Simulation through EpisodeSampler.
        # --------------------------------------------------------------

        self.episode_sampler.start_episode()

        episode_decisions = 0

        # --------------------------------------------------------------
        # Fixed-horizon rollout -> PPO update loop.
        # --------------------------------------------------------------

        while not self.episode_sampler.episode_finished:

            rollout_number = (
                self.episode_sampler.rollout_index
                + 1
            )

            print()
            print(
                f"ROLLOUT {rollout_number}"
            )

            rollout = (
                self.episode_sampler.sample_rollout()
            )

            episode_decisions += (
                len(rollout.transitions)
            )

            # ----------------------------------------------------------
            # PPO update.
            # ----------------------------------------------------------

            update_result = (
                self.ppo_trainer.update(
                    rollout
                )
            )

            self.visualizer.record(
                update_result
            )

            # ----------------------------------------------------------
            # PPO diagnostics.
            # ----------------------------------------------------------

            print(
                "Policy loss:",
                update_result.policy_loss,
            )

            print(
                "Value loss:",
                update_result.value_loss,
            )

            print(
                "Entropy:",
                update_result.entropy,
            )

            print(
                "Approximate KL:",
                update_result.approximate_kl,
            )

            print(
                "Clip fraction:",
                update_result.clip_fraction,
            )

            current_state = (
                self.environment.current_state
            )

            if current_state is not None:

                print(
                    "Throughput:",
                    current_state.throughput,
                )

                print(
                    "Simulation time:",
                    current_state.simulation_time,
                )

        # --------------------------------------------------------------
        # Episode terminal state.
        # --------------------------------------------------------------

        current_state = (
            self.environment.current_state
        )

        if current_state is None:
            raise RuntimeError(
                "Episode terminated without "
                "a current Plant Simulation state."
            )

        succeeded = (
            self.episode_sampler.episode_succeeded
        )

        failed = (
            self.episode_sampler.episode_failed
        )

        self._total_decisions += (
            episode_decisions
        )

        result = EpisodeResult(
            curriculum_stage=stage_index,
            episode_index=episode_index,
            storage_a_capacity=(
                stage.storage_a_capacity
            ),
            storage_b_capacity=(
                stage.storage_b_capacity
            ),
            succeeded=succeeded,
            failed=failed,
            throughput=current_state.throughput,
            simulation_time=current_state.simulation_time,
            rollouts=(
                self.episode_sampler.rollout_index
            ),
            decisions=episode_decisions,
        )

        self.episode_sampler.reset_episode()

        return result

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def _print_episode_result(
        self,
        result: EpisodeResult,
    ) -> None:

        print()
        print(
            "-" * 80
        )

        print(
            "Episode complete"
        )

        print(
            "Stage:",
            result.curriculum_stage,
        )

        print(
            "Stage episode:",
            result.episode_index,
        )

        print(
            "Storage A:",
            result.storage_a_capacity,
        )

        print(
            "Storage B:",
            result.storage_b_capacity,
        )

        print(
            "Succeeded:",
            result.succeeded,
        )

        print(
            "Failed:",
            result.failed,
        )

        print(
            "Throughput:",
            result.throughput,
        )

        print(
            "Simulation time:",
            result.simulation_time,
        )

        print(
            "Rollouts:",
            result.rollouts,
        )

        print(
            "Decisions:",
            result.decisions,
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def total_decisions(self) -> int:
        return self._total_decisions