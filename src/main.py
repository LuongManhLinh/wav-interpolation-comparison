from autoencoder.model import WaveformAutoencoder
from autoencoder.train import train
from autoencoder.evaluate import evaluate

def main():
    model = WaveformAutoencoder(size='medium')
    train(
        model=model,
        data_dir="data/train"
    )

    evaluate(
        model=model,
        data_dir="data/test"
    )
    
if __name__ == "__main__":
    main()