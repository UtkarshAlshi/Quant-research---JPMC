from dataclasses import dataclass

@dataclass
class ContractParams:
    buy_price: float
    sell_price: float
    quantity: float
    storage_fee: float
    storage_duration: int
    injection_withdrawal_cost: float
    transport_cost: float

def calculate_contract_value(params: ContractParams) -> float:
    """
    Calculate the contract value based on given parameters.
    
    :param params: ContractParams object containing all necessary parameters
    :return: Final contract value
    """
    initial_profit = (params.sell_price - params.buy_price) * params.quantity
    total_storage_cost = params.storage_fee * params.storage_duration
    total_transport_cost = params.transport_cost * 2
    
    return (initial_profit 
            - total_storage_cost 
            - params.injection_withdrawal_cost 
            - total_transport_cost)

def main():
    params = ContractParams(
        buy_price=2.0,  # $/MMBtu
        sell_price=3.0,  # $/MMBtu
        quantity=1e6,  # 1 million MMBtu
        storage_fee=100000,  # $100K per month for storage
        storage_duration=4,  # Storing for 4 months
        injection_withdrawal_cost=20000,  # $10K for injection + $10K for withdrawal
        transport_cost=50000  # $50K per trip (2 trips total)
    )
    
    value = calculate_contract_value(params)
    print(f"Final Contract Value: ${value / 1e6:.2f} million")

if __name__ == "__main__":
    main()