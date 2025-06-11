from issue_model import IssueModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the IssueModel")
    parser.add_argument('--input_file', type=str, default='data/mybook2.csv', help="Path to the CSV file for training")
    parser.add_argument('--model_path', type=str, default='search_model.pkl', help="Path to save the trained model")

    args = parser.parse_args()

    model = IssueModel()
    model.train(filepath=args.input_file)  # Pass the custom file path if needed
    model.save(model_path=args.model_path)  # Save to the provided path
