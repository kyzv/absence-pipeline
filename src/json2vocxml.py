# ==========================================
# PHASE 1: LOAD SYSTEM TOOLS
# ==========================================

# Import the 'os' tool to look at folders and files on your computer
import os

# Import the 'json' tool to read LabelMe files
import json

# Import the XML tool and give it the short nickname 'ET'
import xml.etree.ElementTree as ET


# ==========================================
# PHASE 2: DEFINE WHERE YOUR FILES LIVE
# ==========================================

# Set the path to the folder where your LabelMe JSON files are saved
json_dir = "data/annotated/test/"


# ==========================================
# PHASE 3: LOOP THROUGH EVERY FILE
# ==========================================

# Look inside the folder and read the files one by one
for filename in os.listdir(json_dir):
    
    # Check if the current file ends with '.json'. If it does not, ignore it!
    if filename.endswith(".json"):
        
        # Combine the folder path and filename together (e.g., "data/annotated/image1.json")
        full_json_path = os.path.join(json_dir, filename)
        
        # Open the JSON file safely in read-only ("r") mode as a temporary object named 'f'
        with open(full_json_path, "r") as f:
            # Read the file data and convert it into a easy-to-use Python dictionary
            json_data = json.load(f)
            
        # ==========================================
        # PHASE 4: BUILD THE XML MASTER TEMPLATE
        # ==========================================
        
        # Create the outermost XML container tag: <annotation>
        xml_master = ET.Element("annotation")
        
        # Inside <annotation>, create a <filename> tag and fill it with the original image name
        ET.SubElement(xml_master, "filename").text = json_data["imagePath"]
        
        # Inside <annotation>, create a <size> tag to hold dimensions
        xml_size = ET.SubElement(xml_master, "size")
        
        # Put the image width, height, and color channels (3) inside the <size> tag
        # Note: XML requires numbers to be turned into text strings using 'str()'
        ET.SubElement(xml_size, "width").text = str(json_data["imageWidth"])
        ET.SubElement(xml_size, "height").text = str(json_data["imageHeight"])
        ET.SubElement(xml_size, "depth").text = "3"
        
        # ==========================================
        # PHASE 5: PROCESS EACH BOX DRAWN ON THE IMAGE
        # ==========================================
        
        # Loop through every single box shape you drew on this specific image
        for shape in json_data["shapes"]:
            
            # Double check that the shape is actually a rectangle box
            if shape["shape_type"] == "rectangle":
                
                # LabelMe saves rectangles using 2 points: Top-Left (p1) and Bottom-Right (p2)
                p1 = shape["points"][0] # Contains [X-start, Y-start]
                p2 = shape["points"][1] # Contains [X-end, Y-end]
                
                # Turn coordinates into whole integer numbers (no decimals)
                xmin = int(p1[0]) # Top-Left X coordinate (left edge)
                ymin = int(p1[1]) # Top-Left Y coordinate (top edge)
                xmax = int(p2[0]) # Bottom-Right X coordinate (right edge)
                ymax = int(p2[1]) # Bottom-Right Y coordinate (bottom edge)
                
                # Inside <annotation>, create an <object> tag for this specific label
                xml_object = ET.SubElement(xml_master, "object")
                
                # Inside <object>, create a <name> tag containing your label ("header" or "table")
                ET.SubElement(xml_object, "name").text = shape["label"]
                
                # Inside <object>, create a <bndbox> tag to hold the coordinate tags
                xml_bndbox = ET.SubElement(xml_object, "bndbox")
                
                # Drop the 4 boundary edges inside the <bndbox> tag as text strings
                ET.SubElement(xml_bndbox, "xmin").text = str(xmin)
                ET.SubElement(xml_bndbox, "ymin").text = str(ymin)
                ET.SubElement(xml_bndbox, "xmax").text = str(xmax)
                ET.SubElement(xml_bndbox, "ymax").text = str(ymax)
                
        # ==========================================
        # PHASE 6: WRITE THE NEW XML FILE TO HARD DRIVE
        # ==========================================
        
        # Change the filename string from ".json" to ".xml" (e.g., "image1.xml")
        xml_filename = filename.replace(".json", ".xml")
        
        # Combine the folder path and new XML filename together
        full_xml_path = os.path.join(json_dir, xml_filename)
        
        # Compile all the built tags into a finalized document tree
        xml_document = ET.ElementTree(xml_master)
        
        # Save the file permanently to your hard drive
        xml_document.write(full_xml_path)
        
# The script completely finishes once all JSON files have been read and saved as XML.
print("Conversion complete! All JSON files converted to Pascal VOC XML successfully.")
