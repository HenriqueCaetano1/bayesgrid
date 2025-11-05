import pytest


from bayesgrid import BayesianPowerModel, BayesianFrequencyModel, BayesianDurationModel, BayesianImpedanceModel



def test_model_imports():
    bhm = BayesianPowerModel(total_demand=1e3) # 1 GW of total demand

    # 2. Frequency Model
    bfm = BayesianFrequencyModel()

    # 3. Duration Model
    bdm = BayesianDurationModel()

    # 4. Impedance Model (R and X)
    bim = BayesianImpedanceModel()

    
if __name__ == "__main__":
    pytest.main()

