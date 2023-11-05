if __name__ == '__main__':
    import os
    import argparse
    import torch
    from engine.model_builder import Seq2SeqModel
    from engine.engine import train
    from data_setup.data_setup import create_dataloader
    from text_processing.text_transform import preprocess_data, build_text_transform
    from datetime import datetime

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")
    
    # Get an arg for num_epochs
    parser.add_argument("--num_epochs",
                        default=50,
                        type=int,
                        help="the number of epochs to train for")
    
    # Get an arg for learning_rate
    parser.add_argument("--learning_rate",
                        default=0.01,
                        type=float,
                        help="learning rate to use for model")
    
    # Get an arg for teacher forcing ration
    parser.add_argument("--teacher_forcing_ratio",
                        default=0.5,
                        type=float,
                        help="teacher forcing ratio")
    
    # Create an arg for train file path
    parser.add_argument("--train_file_path",
                        #default="./chatbot dataset.txt",
                        default='/home/alex/projects/ml/ml_final_project/seq2seq_module/data/chatbot_dataset.txt',
                        type=str,
                        help="directory file path to training data in standard image classification format")
    
    # Create an arg for sequence max length
    parser.add_argument("--max_length",
                        default=100,
                        type=str,
                        help="directory file path to testing data in standard image classification format")
    
    # Create an arg for target directory
    parser.add_argument("--target_dir",
                        #default='./models',
                        default='/home/alex/projects/ml/ml_final_project/seq2seq_module/models',
                        type=str,
                        help="directory file path to testing data in standard image classification format")

    # Create an arg for target directory
    parser.add_argument("--save_ratio",
                        default=10,
                        type=int,
                        help="number of epochs to save model after")
    
    # Get our arguments from the parser
    args = parser.parse_args()
    
    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
    TRAIN_FILE_PATH = args.train_file_path
    MAX_LENGTH = args.max_length
    TARGET_DIR = args.target_dir
    SAVE_RATIO = args.save_ratio
    HIDDEN_SIZE = 512

    print(
        f"[INFO] Training a model for {NUM_EPOCHS} epochs on data from file '{TRAIN_FILE_PATH}' with learning rate of {LEARNING_RATE}")

    target_dir_name = "training_at_{}".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S'))

    target_dir = TARGET_DIR + "/" + target_dir_name

    # Setup target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reading sentence pairs anb building vocab
    pairs, vocab = preprocess_data(TRAIN_FILE_PATH, MAX_LENGTH)

    os.makedirs(target_dir, exist_ok=True)

    # Save transforms
    torch.save(vocab, "{}/{}".format(target_dir, "vocab.pt"))
    
    # Init text transform
    text_transform = build_text_transform(vocab, MAX_LENGTH)

    # Create dataloader with help from data_setup.py
    dataloader = create_dataloader(pairs, text_transform)

    # Create model with help from model_builder.py
    model = Seq2SeqModel(len(vocab), HIDDEN_SIZE, MAX_LENGTH).to(device)

    # Set loss and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.NLLLoss()

    # Start training with help from engine.py
    train(model, dataloader, loss_fn, optimizer, NUM_EPOCHS, TEACHER_FORCING_RATIO, target_dir, SAVE_RATIO, device)
