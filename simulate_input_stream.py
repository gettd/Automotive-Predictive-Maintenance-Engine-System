import csv
import time

INPUT_CSV = "datasets/synthetic_data/test/obd_21-10-2025_22-45-10-DROP.csv"       
OUTPUT_CSV = "datasets/synthetic_data/temp_output.csv" 

def simulate_stream(input_csv, output_csv, interval=1):
    with open(input_csv, mode="r", newline='') as infile:
        reader = csv.reader(infile)

        header = next(reader)

        with open(output_csv, mode="w", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)

        for row in reader:
            with open(output_csv, mode="a", newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(row)

            print("Wrote row:", row)
            time.sleep(interval) #sleep 1 sec

if __name__ == "__main__":
    simulate_stream(INPUT_CSV, OUTPUT_CSV, interval=1)
