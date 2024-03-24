import open3d as o3d
import numpy as np
import time
import keyboard

class Visu3D():

    def __init__(self):

        self.vis = o3d.visualization.Visualizer()
        self.v = self.vis.create_window()
        self.light = self.vis.get_render_option()
        self.light.light_on = False


        # initial the points information
        self.points = [[[]]]
        self.lines = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6],
                      [7, 8], [8, 9], [8, 11], [8, 14],[9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]

        # set the line color
        self.colors = [[1, 0, 0]] * len(self.lines)

        self.viewCtr = self.vis.get_view_control()
        self.speed = 0.016
        self.curFrame = 0
        self.totalFrame = 0
        self.isStop = False
        self.amp = 12.5
        self.verticalMovement = []

        
        self.ground = o3d.io.read_triangle_mesh("./obj/ground.ply", True)
        self.ground.compute_vertex_normals()

        self.p = o3d.geometry.PointCloud()
        # [x:R, y:G, z:B]
        #self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.line_set = o3d.geometry.LineSet()

    def addPoints(self, predInstance:list):
        self.totalFrame = len(predInstance)

        joints = []

        for p in predInstance:
            joints.append(p[0].tolist())

        index = 0

        for i in range(0, len(joints)):
            
            verticalMove = 0

            if index < len(self.verticalMovement):
                verticalMove = self.verticalMovement[index]
                index += 1

            verticalMove = max(0, verticalMove) 

            for j in range(0, len(joints[i])):
                
                joints[i][j][0] *= self.amp
                joints[i][j][1] *= self.amp

                joints[i][j][2] = joints[i][j][2] * self.amp + verticalMove * 10
  
        self.points = joints


    def addGeometry(self, predInstance:list):

        self.addPoints(predInstance)

        self.vis.add_geometry(self.p)
        #self.vis.add_geometry(self.mesh_frame)
        self.vis.add_geometry(self.line_set)
        self.vis.add_geometry(self.ground)

        self.vis.update_geometry(self.ground)
        self.vis.update_geometry(self.p)
        #self.vis.update_geometry(self.mesh_frame)
        self.vis.update_geometry(self.line_set)

    def speedUp(self):
        self.speed = max(0.008, self.speed - 0.003)

    def speedDown(self):
        self.speed = min(0.16, self.speed + 0.003)

    def run(self):
        self.isStop = False
        while not self.isStop:
            self.nextFrame()
            if not self.vis.poll_events(): return

            if keyboard.is_pressed('z'):
                para = o3d.io.read_pinhole_camera_parameters("./cameraPosition.json")
                print(para)
                self.viewCtr.convert_from_pinhole_camera_parameters(para, True)


            time.sleep(self.speed)

    def stop(self):
        self.isStop = True

    def nextFrame(self):

        """
        update the pose
        """
               
        self.p.points = o3d.utility.Vector3dVector(self.points[self.curFrame])

        self.line_set.points = o3d.utility.Vector3dVector(self.points[self.curFrame])
        self.line_set.lines = o3d.utility.Vector2iVector(self.lines)

        self.line_set.colors = o3d.utility.Vector3dVector(self.colors)
        self.vis.update_geometry(self.p)
        self.vis.update_geometry(self.line_set)

        self.vis.poll_events()
        self.vis.update_renderer()

        self.curFrame = (self.curFrame + 1) % self.totalFrame

    def destroyWindow(self):
        self.isStop = True
        self.vis.destroy_window()

    def setVerticalMovement(self, movement:list):
        self.verticalMovement = movement
