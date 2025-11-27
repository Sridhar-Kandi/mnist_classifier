import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import config
from src.model import MNISTModel

def predict_image(image_path):
    # Load the trained model
    model = MNISTModel().to(config.DEVICE)
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
    model.eval()


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)


    with torch.no_grad():
        output = model(input_tensor)
        probabilities  = torch.nn.functional.softmax(output[0], dim=0)
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item() *  100
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Optional: Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {prediction} ({confidence:.2f}%)")
    plt.show()

if __name__ == "__main__":
    try:
        predict_image('test_images/digit6.jpg') 
    except FileNotFoundError:
        print("Error: Please create or download an image named 'test_digit.png' to test.")
