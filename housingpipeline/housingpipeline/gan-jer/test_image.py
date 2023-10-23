from PIL import Image 


# Make a copy of the graph list
floorplan_ok = False

# Loop till user is satisfied with floorplan
while not floorplan_ok:
    # Open image
    plan = Image.open("/Users/jeroenhofland/Pictures/wapens/wapen_zoelen_lek_m.png")
    plan.show()

    # Check if user is satisfied with floorplan
    response = input("Is this floorplan ok? (yes/no): ").strip().lower()
    if response == "yes":
        floorplan_ok = True
        print("Amazing your floorplan is ready and saved in the output folder.")
    elif response == "no":
        plan.close()
        print("Alright generating a new floorplan.")
    else:
        print("Invalid input. Please enter either 'yes' or 'no'.")