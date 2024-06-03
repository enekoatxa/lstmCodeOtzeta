import bpy
import os

bonesToRemoveFace = ["nose", "nose.001", "nose.002", "nose.003", "nose.004", "lip.T.L", "lip.T.L.001", "lip.B.L", "lip.B.L.001", 
                "jaw", "chin", "chin.001", "ear.L", "ear.L.001", "ear.L.002", "ear.L.003", "ear.L.004",
                "ear.R", "ear.R.001", "ear.R.002", "ear.R.003", "ear.R.004", "lip.T.R", "lip.T.R.001", "lip.B.R", "lip.B.R.001",
                "brow.B.L", "brow.B.L.001", "brow.B.L.002", "brow.B.L.003", "lid.T.L", "lid.T.L.001", "lid.T.L.002", "lid.T.L.003", 
                "lid.B.L", "lid.B.L.001", "lid.B.L.002", "lid.B.L.003",  "brow.B.R", "brow.B.R.001", "brow.B.R.002", "brow.B.R.003", "lid.T.R", 
                "lid.T.R.001", "lid.T.R.002", "lid.T.R.003", "lid.B.R", "lid.B.R.001", "lid.B.R.002", "lid.B.R.003",
                "forehead.L", "forehead.L.001", "forehead.L.002", "temple.L", "jaw.L", "jaw.L.00", "chin.L", "cheek.B.L", "cheek.B.L.001", 
                "brow.T.L", "brow.T.L.001", "brow.T.L.002", "brow.T.L.003", "forehead.R", "forehead.R.001", "forehead.R.002", "temple.R", 
                "jaw.R", "jaw.R.00", "chin.R", "cheek.B.R", "cheek.B.R.001", "brow.T.R", "brow.T.R.001", "brow.T.R.002", "brow.T.R.003",
                "eye.L", "eye.R", "cheek.T.L", "cheek.T.L.001", "nose.L", "nose.L.001", "cheek.T.R", "cheek.T.R.001", "nose.R", "nose.R.001",
                "teeth.T", "teeth.B", "tongue","tongue.001","tongue.002", "jaw.R.001", "jaw.L.001", ]

bonesToRemoveHands = ["palm.01.L", "f_index.01.L", "f_index.02.L", "f_index.03.L", "palm.02.L", "f_middle.01.L", 
                      "f_middle.02.L", "f_middle.03.L", "palm.03.L", "f_ring.01.L", "f_ring.02.L", "f_ring.03.L",
                      "palm.04.L", "f_pinky.01.L", "f_pinky.02.L", "f_pinky.03.L", "thumb.carpal.L", "thumb.01.L",
                      "thumb.02.L", "thumb.03.L", "palm.01.R", "f_index.01.R", "f_index.02.R", "f_index.03.R", 
                      "palm.02.R", "f_middle.01.R", "f_middle.02.R", "f_middle.03.R", "palm.03.R", "f_ring.01.R", 
                      "f_ring.02.R", "f_ring.03.R","palm.04.R", "f_pinky.01.R", "f_pinky.02.R", "f_pinky.03.R", 
                      "thumb.carpal.R", "thumb.01.R", "thumb.02.R", "thumb.03.R", "hand.R", "hand.L"]

heels = ["heel.02.L", "heel.02.R"]

inputBvh = "/home/bee/Desktop/idle animation generator/enekoDatasetNoHandsCen/"
outputBvh = "/home/bee/Desktop/idle animation generator/enekoDatasetNoHandsCen2/"

if not os.path.exists(outputBvh): os.makedirs(outputBvh)

for dirpath, dirnames, filenames in sorted(os.walk(inputBvh)):
    for filename in filenames: 
        '''
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
        '''
        if os.path.basename(filename).split(".")[1]=="bvh":
            bvhFilename = os.path.join(dirpath, filename)
            bpy.ops.import_anim.bvh(filepath=bvhFilename)

            ob = bpy.context.object
            if ob.type == 'ARMATURE':
                armature = ob.data
                bpy.ops.object.mode_set(mode='EDIT')
                for bone in armature.edit_bones:
                    if bone.name in bonesToRemoveFace or bone.name in bonesToRemoveHands or bone.name in heels: 
                        armature.edit_bones.remove(bone)

            bpy.ops.object.mode_set(mode='OBJECT')
            scn = bpy.context.scene
            obAnimation = ob.animation_data.action
            _, frame_end = map(int, obAnimation.frame_range)
            scn.frame_end = frame_end
            
            bpy.ops.export_anim.bvh(filepath=bvhFilename)
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global = False)