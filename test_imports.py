import sys
import os
from pathlib import Path

# Add the vendor directory to path
vendor_path = Path("custom_components/energy_planner/vendor")
if str(vendor_path) not in sys.path:
    sys.path.insert(0, str(vendor_path))

try:
    print("Importing solver...")
    import energy_planner.optimizer.solver as solver
    print("Importing simple_solver...")
    import energy_planner.optimizer.simple_solver as simple_solver
    print("Importing reporting...")
    import energy_planner.reporting as reporting
    print("Importing scheduler...")
    import energy_planner.scheduler as scheduler
    print("All imports successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
