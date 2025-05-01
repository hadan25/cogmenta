@echo off
echo Setting up ConceptNet Training Environment...

REM Create required directories
if not exist training\datasets\conceptnet mkdir training\datasets\conceptnet
if not exist conceptnet_training_output mkdir conceptnet_training_output

REM Check if sample data exists
if not exist training\datasets\conceptnet\sample_data.csv (
    echo Creating sample data...
    
    REM Create sample data
    echo relation	subject	object	weight > training\datasets\conceptnet\sample_data.csv
    echo IsA	dog	animal	5.0 >> training\datasets\conceptnet\sample_data.csv
    echo IsA	cat	animal	5.0 >> training\datasets\conceptnet\sample_data.csv
    echo IsA	tree	plant	4.5 >> training\datasets\conceptnet\sample_data.csv
    echo HasProperty	dog	loyal	3.8 >> training\datasets\conceptnet\sample_data.csv
    echo HasProperty	cat	independent	3.6 >> training\datasets\conceptnet\sample_data.csv
    echo CapableOf	dog	bark	4.2 >> training\datasets\conceptnet\sample_data.csv
    echo CapableOf	bird	fly	4.7 >> training\datasets\conceptnet\sample_data.csv
    echo UsedFor	car	transportation	4.9 >> training\datasets\conceptnet\sample_data.csv
    echo AtLocation	fish	water	4.5 >> training\datasets\conceptnet\sample_data.csv
    echo AtLocation	book	library	3.8 >> training\datasets\conceptnet\sample_data.csv
    echo HasA	car	engine	4.6 >> training\datasets\conceptnet\sample_data.csv
    echo HasA	human	brain	5.0 >> training\datasets\conceptnet\sample_data.csv
    echo PartOf	leaf	tree	4.2 >> training\datasets\conceptnet\sample_data.csv
    echo Causes	rain	wetness	4.7 >> training\datasets\conceptnet\sample_data.csv
    echo Causes	hunger	eating	4.8 >> training\datasets\conceptnet\sample_data.csv
    echo Causes	studying	knowledge	4.3 >> training\datasets\conceptnet\sample_data.csv
)

echo.
echo ConceptNet Training Environment Setup Complete!
echo.
echo You can now run the training using one of the following methods:
echo.
echo   1. Interactive menu (recommended):
echo      run_conceptnet_training.bat
echo.
echo   2. Basic training (skips integration phase):
echo      python run_conceptnet.py
echo.
echo   3. Full training with Prolog fixes:
echo      python run_fixed_conceptnet.py
echo.
echo   4. Load sample data only (no training):
echo      python load_conceptnet.py
echo.
echo   5. Run with real ConceptNet data:
echo      python -m training.run_concept_training
echo.
echo Alternatively, use this command to run the trainer in your Python code:
echo.
echo   from training.trainers.concept_net_trainer import ConceptNetTrainer
echo   trainer = ConceptNetTrainer(output_dir="output")
echo   results = trainer.run_full_training()
echo.

# Run the full training pipeline
python -m training.quickstart_training

# Run specific trainers
python -m training.quickstart_training --training_phases=atomic,nli,logiqa 