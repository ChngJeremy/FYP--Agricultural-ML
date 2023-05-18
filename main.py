import time
from sensor import collect_sensor_data

# Collection interval in seconds
DATA_COLLECTION_INTERVAL = 5

# Main function
def main():
    while True:
        # Collect sensor data
        entropy = collect_sensor_data()

        # Print entropy value
        print(f"Entropy: {entropy}")
        print("-------------------------")

        # Delay before next iteration
        time.sleep(DATA_COLLECTION_INTERVAL)

# Execute the main function
if __name__ == '__main__':
    main()