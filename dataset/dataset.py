import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class EasyQADataset(Dataset):
    """
    A custom dataset class for handling vision-language data.
    It loads the data entries, applies necessary transformations, and prepares inputs for the model.
    """
    def __init__(self, df, tokenizer, image_transform=None, device='cpu'):
        """
        Initializes the dataset with necessary transformations and components.

        Args:
            df (DataFrame): DataFrame containing the data entries.
            tokenizer (callable): Tokenizer function for text processing.
            image_transform (callable, optional): Transformations to be applied to images.
            device (str): The device ('cuda' or 'cpu') where tensors will be sent.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.device = device

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the indexed data and returns it in a processed form suitable for model input.

        Args:
            idx (int): The index of the data entry to retrieve.

        Returns:
            dict: A dictionary containing processed image and text data, along with the label.
        """
        row = self.df.iloc[idx]
        image_path = row['image_path']
        question = row['question']
        label = row['label']

        # Load the image and apply transformations
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # Prepare text input
        text_inputs = self.tokenizer(question, return_tensors='pt')
        text_input_ids = text_inputs['input_ids'].squeeze(0)  # Remove batch dimension if it exists

        # Create a dictionary to hold the processed data
        sample = {
            'image': image.to(self.device),
            'text': text_input_ids.to(self.device),
            'label': torch.tensor(label, dtype=torch.long).to(self.device)
        }

        return sample
