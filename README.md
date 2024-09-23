To design the modular code for your model based on **TCCT-Net** architecture as detailed in the paper you provided【6†source】, we will follow a pipeline design that includes several key steps. Here’s the pipeline design with detailed steps for training, testing, and validation:

### **Pipeline Design for TCCT-Net Model**

1. **Preprocessing (Feature Extraction with OpenFace)**
    - Extract behavioral signals (Action Units (AUs), Eye Gaze, Head Pose) from video frames.
    - Use `OpenFace` feature extractor script (already provided in your OpenFace extraction script)【5†source】. Each signal represents a time series of values for a specific extracted feature.
    - Stack these signals from the video to create a 2D matrix (temporal-spatial matrix) representing the behavioral features and time series.

2. **Data Preparation**
    - Group the dataset by subject and video, ensuring all data is properly labeled for training, validation, and testing splits.
    - Normalize the behavioral feature signals, ensuring the data follows a uniform scale across different features and videos.
    - For each subject, split the data into the input matrix (F signals) and the target labels (engagement levels).

3. **Data Augmentation**
    - Implement the **Segmentation and Recombination (S&R)** technique for data augmentation【6†source】. This technique generates new augmented samples by segmenting and recombining behavioral signal data.
    - Maintain data consistency and introduce realistic variations during training, avoiding overfitting.

4. **Model Architecture Design**
    - **Temporal-Spatial Stream (CT Stream)**:
      - Utilize **Conformer architecture** to process the temporal-spatial feature matrix.
      - Incorporate convolutional layers to extract local features from the temporal-spatial data.
      - Follow this with multi-head self-attention to capture global dependencies.
      - Apply a dense classifier layer to produce an intermediate engagement score for the stream.
    
    - **Temporal-Frequency Stream (TC Stream)**:
      - Convert behavioral signals into time-frequency tensors using **Continuous Wavelet Transform (CWT)**.
      - Use 2D tensors for each behavioral feature signal.
      - Apply convolutional layers to extract important temporal-frequency features, followed by pooling and dense layers.
    
    - **Fusion Layer**:
      - After processing in both streams, fuse the results using a weighted decision fusion mechanism.
      - The final output combines the engagement predictions from both streams.

5. **Loss Function and Optimization**
    - Use **Cross-Entropy Loss** to measure prediction error.
    - Add **L2 Regularization** to penalize complex models, preventing overfitting.
    - Implement **Adam Optimizer** with learning rate scheduling.

6. **Training and Validation Loop**
    - Train the model using **subject-independent** splits (train on one group of subjects, validate on another, test on the final group).
    - Implement early stopping based on validation performance to prevent overfitting.
    - Periodically evaluate model performance using accuracy, F1-score, and inference time.
    
7. **Testing**
    - After training, run inference on the test set.
    - Measure the test accuracy and compare the model's efficiency (inference time) with the baseline methods provided.

---

### **Detailed Modular Steps**

0. **Project Structure**
1. Parameters and Hyperparameters can be changed in utility/config.py
```Iua
    Project Root/
    ├── config.py               # Global configuration instance
    ├── main.py                 # Entry point script
    ├── utils/
    │   ├── __init__.py
    │   ├── logger.py
    │   ├── config.py           # Contains the updated Config class
    │   ├── seed.py
    │   ├── metrics.py
    │   └── visualization.py
    ├── models/
    │   ├── __init__.py
    │   ├── conformer.py
    │   ├── ct_stream.py
    │   ├── tc_stream.py
    │   └── tcct_net.py
    ├── dataset_processing/
    │   ├── __init__.py
    │   ├── data_preprocessing.py
    │   ├── data_augmentation.py
    │   ├── data_loading.py
    │   └── collate_fn.py
    ├── training/
    │   ├── __init__.py
    │   ├── train.py
    │   ├── trainer.py
    │   ├── evaluator.py
    │   ├── metrics.py
    │   └── checkpoint.py
    ├── logs/
    │   ├── training.log
    │   ├── data_preprocessing.log
    │   ├── data_augmentation.log
    │   ├── data_loading.log
    │   └── evaluation.log
    ├── data/
    │   ├── processed_dataset/
    │   ├── train_engagement_labels.xlsx
    │   └── ... (other data files)
    ├── requirements.txt
    └── README.md

```

1. **Preprocessing**
    ```python
    # main.py

    import torch
    from torch import nn, optim
    from prepare_data_module import prepare_data

    def main():
        # Configuration variables
        folder_path = 'processed_dataset'
        label_file = 'train_engagement_labels.xlsx'
        label_column = 'engagement_level'
        exclude_columns = ['frame', 'timestamp']
        missing_value_strategy = 'interpolate'
        segment_length = 50
        num_augmented_samples = 1
        test_size = 0.2
        batch_size = 32
        num_workers = 4
        random_state = 42
        num_epochs = 50

        # Prepare data
        train_loader, val_loader, train_dataset, val_dataset = prepare_data(
            folder_path,
            label_file,
            label_column,
            exclude_columns,
            missing_value_strategy,
            segment_length,
            num_augmented_samples,
            test_size,
            batch_size,
            num_workers,
            random_state
        )

        # Access dataset details
        train_dataset.describe()
        val_dataset.describe()

        # Define your model
        model = MyModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for data, labels, sequence_lengths in train_loader:
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, labels, sequence_lengths in val_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * data.size(0)
            val_loss /= len(val_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Save the model
        torch.save(model.state_dict(), 'model.pth')
    ```

2. **Augmentation**
    ```python
    # main_preprocessing.py

    import os
    import logging
    # You can import directly from the dataset_processing package:
    from dataset_processing import EngagementDataset, collate_fn


    def main():
        # Configure logging
        logging.basicConfig(
            filename='data_preprocessing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Configuration variables
        folder_path = 'processed_dataset'  # Directory containing CSV files
        label_file = 'train_engagement_labels.xlsx'  # Path to labels file
        label_column = 'engagement_level'
        exclude_columns = ['frame', 'timestamp']  # Exclude non-feature columns
        missing_value_strategy = 'interpolate'  # Strategy for missing values
        save_dir = 'preprocessed_data'  # Directory to save preprocessed data

        # Load file paths and labels (assuming you have a function for this)
        from data_loading_module import load_csv_data
        file_paths, labels, _ = load_csv_data(
            folder_path, label_file, label_column, exclude_columns
        )

        # Preprocess data
        features_list, labels, scaler = preprocess_data(
            file_paths, labels, exclude_columns=exclude_columns,
            missing_value_strategy=missing_value_strategy, save_dir=save_dir
        )


    ```
    ```python
    # main_augmentation.py

    import logging
    from data_augmentation import augment_dataset

    def main():
        # Configure logging
        logging.basicConfig(
            filename='data_augmentation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Configuration variables
        segment_length = 50  # Adjust based on your data
        num_augmented_samples = 2  # Number of augmented samples per original sample

        # Load preprocessed data (assuming you have a function or method for this)
        from data_preprocessing import preprocess_data
        # Use the same preprocessing steps to get features_list and labels
        # Or load from saved preprocessed data
        # features_list, labels = load_preprocessed_data('preprocessed_data')

        # For demonstration, let's assume features_list and labels are already loaded
        features_list = [...]  # List of np.ndarray
        labels = [...]  # List of int

        # Augment dataset
        augmented_features_list, augmented_labels = augment_dataset(
            features_list, labels, segment_length, num_augmented_samples
        )

        # Now you can proceed to use augmented_features_list and augmented_labels for training
    
    ```

3. **Model Architecture (TCCT-Net)**
    ```python
    # example_usage.py

    import torch
    from config import Config
    from models import CTStream, TCStream, TCCTNet


    def main():
        """
        Example usage of the TCCT-Net model.
        This script demonstrates how to initialize the model, create sample input data,
        and perform a forward pass to obtain output logits.
        """
        # ----------------------------
        # Initialize Configuration
        # ----------------------------
        config = Config()
        config.set_seed(config.random_seed)  # Set random seed for reproducibility

        # ----------------------------
        # Create Sample Input Data
        # ----------------------------
        batch_size = 2  # Example batch size
        seq_length = 100  # Example sequence length (can be any integer <= config.max_sequence_length)
        input_dim = config.input_dim  # Should match the input_dim in Config

        # Generate random input data (batch_size, seq_length, input_dim)
        sample_input = torch.randn(batch_size, seq_length, input_dim, device=config.device)

        # Create sequence lengths tensor (all sequences have the same length in this example)
        seq_lengths = torch.tensor([seq_length] * batch_size, dtype=torch.long, device=config.device)

        # ----------------------------
        # Initialize the Model
        # ----------------------------
        model = TCCTNet(config).to(config.device)  # Move model to the configured device

        # Set model to evaluation mode (since we're not training)
        model.eval()

        # ----------------------------
        # Perform Forward Pass
        # ----------------------------
        with torch.no_grad():  # Disable gradient calculation for inference
            output_logits = model(sample_input, seq_lengths)  # Obtain output logits

        # ----------------------------
        # Inspect the Output
        # ----------------------------
        print("Output logits shape:", output_logits.shape)  # Should be (batch_size, num_classes)
        print("Output logits:", output_logits)

        # Optionally, apply softmax to get probabilities
        probabilities = torch.softmax(output_logits, dim=1)
        print("Output probabilities:", probabilities)

        # Get predicted classes
        predicted_classes = torch.argmax(output_logits, dim=1)
        print("Predicted classes:", predicted_classes)

    ```

4. **Training and Validation Loop**
    ```python
    def train_model(model, train_data, val_data, epochs=50):
        # Compile model with Adam optimizer and loss function
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stopping])

        return history
    ```

5. **Training Setup**
    To run your script on multiple GPUs using PyTorch **Distributed Data Parallel (DDP)**, we used the `torch.distributed.launch` or `torchrun` utility.

    ### Using **`torchrun`**,(PyTorch >=1.9.0):

    ```bash
    torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE main.py
    ```

    ### Using `torch.distributed.launch`:

    ```bash
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE main.py
    ```

    ### **Multi-GPU Support**: Need to wrap model in `torch.nn.parallel.DistributedDataParallel`.

    Make sure `CUDA_VISIBLE_DEVICES` is correctly set if you only want to use certain GPUs
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py
    ```

    **Ensure `local_rank` Handling**: When using DDP, each process needs to know which GPU it is assigned to, which is passed through the `--local_rank` argument.


This pipeline closely follows the steps outlined in the TCCT-Net architecture, while also ensuring modularity for scalability and maintainability.
