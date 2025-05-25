from GUITrainer import GUITrainer
from GaussianTrainer import GaussianTrainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="text prompt for training or generation")
    args = parser.parse_args()
    parser.add_argument("--gui", action="store_true", help="enable GUI")
    print(f"Prompt received: {args.prompt}")

    if args.gui:
        gui = GUITrainer(args.prompt)
        gui.run()
    else:
        non_gui = GaussianTrainer(args.prompt)
        non_gui.train()
