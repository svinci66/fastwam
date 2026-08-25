"""Train the no-Q residual AWR actor from a RoboTwin replay.

The replay schema and learner are benchmark-agnostic, so this entrypoint reuses
the validated residual-AWR implementation.  It exists to give RoboTwin runs an
unambiguous command and to prevent accidental use of the legacy IQL trainer.
"""

from train_libero_residual_awr import main


if __name__ == "__main__":
    main()
