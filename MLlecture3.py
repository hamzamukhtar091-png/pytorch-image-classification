import os
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)


# ============================================================
# 2. PATHS
# ============================================================

DATASET_PATH = r"C:\Users\Al Shahraz Mobile\Downloads\Dogs Vs Cats"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

print("Train path:", TRAIN_PATH)
print("Test path:", TEST_PATH)


# ============================================================
# 3. FIND ALL IMAGES
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


def find_images(folder):

    images = []

    for root, folders, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(image_extensions):

                full_path = os.path.join(root, filename)

                images.append(full_path)

    return images


all_train_images = find_images(TRAIN_PATH)

print("\nTotal images found in train:", len(all_train_images))


# ============================================================
# 4. FIND CATS AND DOGS
# ============================================================

cat_images = []
dog_images = []


for image_path in all_train_images:

    filename = os.path.basename(image_path).lower()

    # Check filename
    if "cat" in filename:

        cat_images.append(image_path)

    elif "dog" in filename:

        dog_images.append(image_path)


print("Cats found:", len(cat_images))
print("Dogs found:", len(dog_images))


# ============================================================
# 5. CHECK DATASET
# ============================================================

if len(cat_images) == 0 or len(dog_images) == 0:

    print("\nERROR!")
    print("Python could not find both cats and dogs.")
    print("\nFirst 20 files found:")

    for image in all_train_images[:20]:

        print(os.path.basename(image))

    input("\nPress Enter to close...")

    raise SystemExit


# ============================================================
# 6. BALANCE DATASET
# ============================================================

random.seed(42)

random.shuffle(cat_images)
random.shuffle(dog_images)


MAX_EACH_CLASS = 4000


cat_images = cat_images[:MAX_EACH_CLASS]

dog_images = dog_images[:MAX_EACH_CLASS]


print("\nUsing dataset:")

print("Cats:", len(cat_images))

print("Dogs:", len(dog_images))

print("Total:", len(cat_images) + len(dog_images))


# ============================================================
# 7. CREATE DATA + LABELS
# ============================================================

all_data = []


# CAT = 0

for image_path in cat_images:

    all_data.append(
        (image_path, 0)
    )


# DOG = 1

for image_path in dog_images:

    all_data.append(
        (image_path, 1)
    )


# Shuffle

random.shuffle(all_data)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(all_data) * 0.8
)


train_data = all_data[:split_index]

val_data = all_data[split_index:]


print("\nDataset split:")

print(
    "Training images:",
    len(train_data)
)

print(
    "Validation images:",
    len(val_data)
)


# ============================================================
# 9. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# ============================================================
# 10. CUSTOM DATASET
# ============================================================

class CatsDogsDataset(Dataset):

    def __init__(
        self,
        data,
        transform=None
    ):

        self.data = data

        self.transform = transform


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        image_path, label = self.data[index]

        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(image)


        return image, label


# ============================================================
# 11. CREATE DATASETS
# ============================================================

train_dataset = CatsDogsDataset(
    train_data,
    train_transform
)


val_dataset = CatsDogsDataset(
    val_data,
    val_transform
)


print("\nDatasets created!")


# ============================================================
# 12. DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False,

    num_workers=0
)


print("Data loaders ready!")


# ============================================================
# 13. CNN MODEL
# ============================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CNN().to(device)


print("\nCNN model created!")


# ============================================================
# 14. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001
)


# ============================================================
# 15. TRAINING
# ============================================================

EPOCHS = 5


print("\n")
print("==============================")
print("STARTING TRAINING")
print("==============================")


for epoch in range(EPOCHS):


    # ----------------------------
    # TRAIN
    # ----------------------------

    model.train()


    total_loss = 0

    correct = 0

    total = 0


    for batch_number, (images, labels) in enumerate(train_loader):


        images = images.to(device)

        labels = labels.to(device)


        # Remove old gradients

        optimizer.zero_grad()


        # Model prediction

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        total_loss += loss.item()


        # Get prediction

        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


        # Progress

        if (batch_number + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch_number + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )


    train_accuracy = (
        100 * correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:


            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)


            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    average_loss = (
        total_loss / len(train_loader)
    )


    print("\n--------------------------------")

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Loss: {average_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print("--------------------------------")


# ============================================================
# 16. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "cat_dog_cnn.pth"
)


print("\nModel saved!")

print(
    "File: cat_dog_cnn.pth"
)


# ============================================================
# 17. FIND TEST IMAGES
# ============================================================

test_images = find_images(TEST_PATH)


print(
    "\nTest images found:",
    len(test_images)
)


# ============================================================
# 18. SHOW TEST PREDICTIONS
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


random.shuffle(test_images)


sample_images = test_images[:8]


model.eval()


plt.figure(
    figsize=(12, 8)
)


for i, image_path in enumerate(sample_images):


    image = Image.open(
        image_path
    ).convert("RGB")


    input_image = test_transform(
        image
    )


    input_image = input_image.unsqueeze(0)


    input_image = input_image.to(device)


    with torch.no_grad():

        output = model(
            input_image
        )


        probability = torch.softmax(
            output,
            dim=1
        )


        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()


        confidence = (
            probability[0][predicted_class].item()
            * 100
        )


    if predicted_class == 0:

        prediction = "CAT"

    else:

        prediction = "DOG"


    plt.subplot(
        2,
        4,
        i + 1
    )


    plt.imshow(image)


    plt.title(
        f"{prediction}\n"
        f"{confidence:.1f}%"
    )


    plt.axis("off")


plt.tight_layout()

plt.show()


print("\n==============================")
print("DONE!")
print("==============================")