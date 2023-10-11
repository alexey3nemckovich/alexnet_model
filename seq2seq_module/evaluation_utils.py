import random
from text_transform import text_encode, text_decode


def evaluateRandomly(model, vocab, text_transform, pairs, n, device):
    for i in range(n):
        pair = random.choice(pairs)

        translated_sentence = text_decode(model.forward_inference(text_encode(pair[0], text_transform).to(device), device), vocab)

        print('>', pair[0])
        print('=', pair[1])
        print('<', translated_sentence)
        print('')
