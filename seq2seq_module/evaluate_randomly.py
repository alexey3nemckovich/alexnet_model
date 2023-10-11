if __name__ == '__main__':
    import argparse
    import torch
    from model_builder import Seq2SeqModel
    from utils import load_model
    from text_transform import build_text_transform
    from file_utils import read_pairs
    from evaluation_utils import evaluateRandomly

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")
    
    # Create an arg for train file path
    parser.add_argument("--model_file_path",
                        #default="./chatbot dataset.txt",
                        default='/home/alex/projects/ml/ml_final_project/seq2seq_module/models/model.pth',
                        type=str,
                        help="directory file path to training data in standard image classification format")
    

    # Create an arg for train file path
    parser.add_argument("--vocab_file_path",
                        #default="./chatbot dataset.txt",
                        default='/home/alex/projects/ml/ml_final_project/seq2seq_module/models/vocab.pt',
                        type=str,
                        help="directory file path to training data in standard image classification format")
    
    # Create an arg for train file path
    parser.add_argument("--data_file_path",
                        #default="./chatbot dataset.txt",
                        default='/home/alex/projects/ml/ml_final_project/seq2seq_module/data/chatbot_dataset.txt',
                        type=str,
                        help="directory file path to training data in standard image classification format")
    
    # Get an arg for num_epochs
    parser.add_argument("--n",
                        default=10,
                        type=int,
                        help="the number of epochs to train for")
    
    # Get our arguments from the parser
    args = parser.parse_args()
    
    # Setup hyperparameters
    MODEL_FILE_PATH = args.model_file_path
    VOCAB_FILE_PATH = args.vocab_file_path
    DATA_FILE_PATH = args.data_file_path
    N = args.n

    print(
        f"[INFO] Evaluating a model from file {MODEL_FILE_PATH} with data from {DATA_FILE_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = torch.load(VOCAB_FILE_PATH)

    model = Seq2SeqModel(len(vocab), 512, 100).to(device)

    load_model(model, MODEL_FILE_PATH)

    model = model.to(device)

    text_transform = build_text_transform(vocab, model.max_length)

    pairs = read_pairs(DATA_FILE_PATH)

    evaluateRandomly(model, vocab, text_transform, pairs, N, device)
    
