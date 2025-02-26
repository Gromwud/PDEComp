from code.src.data_loader import *
from code.src.wrappers import *


# Usage example
if __name__ == "__main__":
    # Load your data
    u, x, t = load_data()  # You need to implement this function

    # Create wrappers
    pde_find = PDEFINDWrapper()
    wsindy = WSINDyWrapper()

    # Run PDE discovery
    pde_find_result = pde_find.run(u, x, t)
    wsindy_result = wsindy.run(u, x, t)

    # Compare results
    print("PDE-FIND result:", pde_find_result)
    print("WSINDy result:", wsindy_result)

