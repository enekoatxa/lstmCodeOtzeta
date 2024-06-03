import bpy
import os

# execute script one time to rotate 90 degrees
inputBvh = "/home/bee/Desktop/idle animation generator/enekoDatasetNoHandsCen2/"
outputBvh = "/home/bee/Desktop/idle animation generator/enekoDatasetNoHandsCen2/"

if not os.path.exists(outputBvh): os.makedirs(outputBvh)

for dirpath, dirnames, filenames in sorted(os.walk(inputBvh)):
    for filename in filenames: 
        if("idle." in filename):
            exportFilename = os.path.join(outputBvh, "idle")
            bvhFilename = os.path.join(inputBvh, "idle", filename)
            if not os.path.exists(exportFilename): os.makedirs(exportFilename)
            exportFilename = os.path.join(exportFilename, filename)
        if("idle2." in filename):
            exportFilename = os.path.join(outputBvh, "idle2")
            bvhFilename = os.path.join(inputBvh, "idle2", filename)
            if not os.path.exists(exportFilename): os.makedirs(exportFilename)
            exportFilename = os.path.join(exportFilename, filename)
        if("actions." in filename):
            exportFilename = os.path.join(outputBvh, "actions")
            bvhFilename = os.path.join(inputBvh, "actions", filename)
            if not os.path.exists(exportFilename): os.makedirs(exportFilename)
            exportFilename = os.path.join(exportFilename, filename)
        if("phone." in filename):
            exportFilename = os.path.join(outputBvh, "phone")
            bvhFilename = os.path.join(inputBvh, "phone", filename)
            if not os.path.exists(exportFilename): os.makedirs(exportFilename)
            exportFilename = os.path.join(exportFilename, filename)
        if("lookback." in filename):
            exportFilename = os.path.join(outputBvh, "lookback")
            bvhFilename = os.path.join(inputBvh, "lookback", filename)
            if not os.path.exists(exportFilename): os.makedirs(exportFilename)
            exportFilename = os.path.join(exportFilename, filename)

        if os.path.basename(filename).split(".")[1]=="bvh":
            print(bvhFilename)
            print(exportFilename)
            bpy.ops.import_anim.bvh(filepath=bvhFilename)

            ob = bpy.context.object
            bpy.ops.object.mode_set(mode='OBJECT')
            scn = bpy.context.scene
            obAnimation = ob.animation_data.action
            _, frame_end = map(int, obAnimation.frame_range)
            scn.frame_end = frame_end
            
            bpy.ops.export_anim.bvh(filepath=exportFilename)
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global = False)