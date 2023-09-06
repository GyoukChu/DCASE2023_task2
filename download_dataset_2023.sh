# If " syntax error near unexpected token `$'{\r'' " error occurs, do
# sed -i 's/\r//' download_dataset_2023.sh

# download dev data
mkdir -p "data/train"
cd "data/train"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/7882613/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data - train
for machine_type in bandsaw grinder shaker ToyDrone ToyNscale ToyTank Vacuum; do
wget "https://zenodo.org/record/7830345/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"
done

cd ..
cd ..

# download eval data - test
mkdir -p "data/test"
cd "data/test"
for machine_type in bandsaw grinder shaker ToyDrone ToyNscale ToyTank Vacuum; do
wget "https://zenodo.org/record/7860847/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done