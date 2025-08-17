import torch
import huggingface_hub
import transformers
import datasets
import bitsandbytes
import peft

print("test")

# Check if GPU is available
if torch.cuda.is_available():
    print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Run a small test on the GPU
try:
    # Create a random tensor and send it to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.rand(10000, 10000, device=device)
    print("Tensor created on device:", tensor.device)

    # Perform a simple computation
    result = tensor @ tensor.T  # Matrix multiplication
    print("Matrix multiplication result shape:", result.shape)
    print("Computation successful on GPU!")

except RuntimeError as e:
    print(f"Error during computation: {e}")

print("torch version")
print(torch.__version__)  # Should print 2.5.0 or similar
print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)  # Should print 12.8


#hugging face test
print("hugging face version")
print(huggingface_hub.__version__)  # Should print 0.26.0
print(torch.cuda.is_available())

#transformer tests
print("transformer version")
print(transformers.__version__)  # Should print 4.44.2
print(torch.cuda.is_available())  # Should still print True

#datasets (for tweets)
print("datasets version")
print(datasets.__version__)  # Should print 3.0.1
print(torch.cuda.is_available())  # Should still print True

#bitsandbytets
print("bitsandbytes version")
print(bitsandbytes.__version__)  # Should print 0.44.0
print(torch.cuda.is_available())  # Should still print True



#peft
print("peft version")
print(peft.__version__)  # Should print 0.13.2
print(torch.cuda.is_available())  # Should still print True
