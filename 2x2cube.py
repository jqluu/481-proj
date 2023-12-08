# sys
import time
import sys

# displayCuber
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# bfs
from collections import deque

# dfs testing
sys.setrecursionlimit(10**6)

class RubiksCube2x2:
    def __init__(self):
        # initial state = solved cube
        self.up_face = [[0, 0], [0, 0]]       # white
        self.front_face = [[1, 1], [1, 1]]   # red
        self.right_face = [[2, 2], [2, 2]]   # green
        self.left_face = [[3, 3], [3, 3]]    # orange
        self.back_face = [[4, 4], [4, 4]]    # blue
        self.down_face = [[5, 5], [5, 5]]    # yellow

    def is_solved(self):
        return (
            self.up_face == [[0, 0], [0, 0]] and
            self.front_face == [[1, 1], [1, 1]] and
            self.right_face == [[2, 2], [2, 2]] and
            self.left_face == [[3, 3], [3, 3]] and
            self.back_face == [[4, 4], [4, 4]] and
            self.down_face == [[5, 5], [5, 5]]
        )

    def copy(self):
        # return a copy (new obj)
        copied_cube = RubiksCube2x2()
        copied_cube.up_face = [row[:] for row in self.up_face]
        copied_cube.front_face = [row[:] for row in self.front_face]
        copied_cube.right_face = [row[:] for row in self.right_face]
        copied_cube.left_face = [row[:] for row in self.left_face]
        copied_cube.back_face = [row[:] for row in self.back_face]
        copied_cube.down_face = [row[:] for row in self.down_face]
        return copied_cube

    def apply_move(self, move):
        # given face, rotate clockwise
        if move == 'R':
            self.right_face = self.rotate_clockwise(self.right_face)
            temp = [self.front_face[0][1], self.front_face[1][1]]  
            self.front_face[0][1], self.front_face[1][1] = self.down_face[0][1], self.down_face[1][1]
            self.down_face[0][1], self.down_face[1][1] = self.back_face[1][0], self.back_face[0][0]
            self.back_face[1][0], self.back_face[0][0] = self.up_face[0][1], self.up_face[1][1]
            self.up_face[0][1], self.up_face[1][1] = temp[0], temp[1]
        
        elif move == 'L':
            self.left_face = self.rotate_clockwise(self.left_face)
            temp = [self.front_face[0][0], self.front_face[1][0]]
            self.front_face[0][0], self.front_face[1][0] = self.up_face[0][0], self.up_face[1][0]
            self.up_face[0][0], self.up_face[1][0] = self.back_face[1][1], self.back_face[0][1]
            self.back_face[1][1], self.back_face[0][1] = self.down_face[0][0], self.down_face[1][0]
            self.down_face[0][0], self.down_face[1][0] = temp[0], temp[1]
        
        elif move == 'U':
            self.up_face = self.rotate_clockwise(self.up_face)
            temp = [self.front_face[0][0], self.front_face[0][1]]
            self.front_face[0][0], self.front_face[0][1] = self.right_face[0][0], self.right_face[0][1]
            self.right_face[0][0], self.right_face[0][1] = self.back_face[0][0], self.back_face[0][1]
            self.back_face[0][0], self.back_face[0][1] = self.left_face[0][0], self.left_face[0][1]
            self.left_face[0][0], self.left_face[0][1] = temp[0], temp[1]
        
        elif move == 'D':
            self.down_face = self.rotate_clockwise(self.down_face)
            temp = [self.front_face[1][0], self.front_face[1][1]]
            self.front_face[1][0], self.front_face[1][1] = self.left_face[1][0], self.left_face[1][1]
            self.left_face[1][0], self.left_face[1][1] = self.back_face[1][0], self.back_face[1][1]
            self.back_face[1][0], self.back_face[1][1] = self.right_face[1][0], self.right_face[1][1]
            self.right_face[1][0], self.right_face[1][1] = temp[0], temp[1]
        
        elif move == 'F':
            self.front_face = self.rotate_clockwise(self.front_face)
            temp = [self.up_face[1][0], self.up_face[0][0]]
            self.up_face[1][0], self.up_face[0][0] = self.left_face[1][1], self.left_face[0][1]
            self.left_face[1][1], self.left_face[0][1] = self.down_face[0][1], self.down_face[1][1]
            self.down_face[0][1], self.down_face[1][1] = self.right_face[0][0], self.right_face[1][0]
            self.right_face[0][0], self.right_face[1][0] = temp[0], temp[1]
        
        elif move == 'B':
            self.back_face = self.rotate_clockwise(self.back_face)
            temp = [self.up_face[0][1], self.up_face[1][1]]
            self.up_face[0][1], self.up_face[1][1] = self.right_face[0][1], self.right_face[1][1]
            self.right_face[0][1], self.right_face[1][1] = self.down_face[1][1], self.down_face[0][1]
            self.down_face[1][1], self.down_face[0][1] = self.left_face[1][0], self.left_face[0][0]
            self.left_face[1][0], self.left_face[0][0] = temp[0], temp[1]

    
    # rotation func
    def rotate_clockwise(self, face):
        return [
            [face[1][0], face[0][0]],
            [face[1][1], face[0][1]]
        ]

    # def rotate_counterclockwise(self, face):
    #     return [
    #         [face[0][1], face[1][1]],
    #         [face[0][0], face[1][0]]
    #     ]
    
    # for testing
    def apply_multMoves(self, moves):
        for move in moves:
            self.apply_move(move)
    

    # plot 
    def displayCube(self):
        colors = ['white', 'red', 'green', 'orange', 'blue', 'yellow']
        color_map = ListedColormap(colors)

        faces = [
            self.up_face, self.front_face, self.right_face,
            self.left_face, self.back_face, self.down_face
        ]

        fig, axs = plt.subplots(2, 3, figsize=(6, 4))

        for i, ax in enumerate(axs.flat):
            face = faces[i]
            ax.imshow(face, cmap=color_map, vmin=0, vmax=5)
            ax.axis('off')
            ax.set_title(['Up', 'Front', 'Right', 'Left', 'Back', 'Down'][i])

        plt.tight_layout()
        plt.show()
    

# bfs
class bfs:
    def __init__(self):
        self.cube = RubiksCube2x2()
        self.moves = ['R', 'L', 'U', 'D', 'F', 'B']  
        
    def solve(self):
        queue = deque([(self.cube.copy(), [])])  # queue
        visited = set()  # visited

        while queue:
            current, path = queue.popleft()
            if current.is_solved():
                return path  

            cube_state = tuple(tuple(row) for face in [
                current.up_face, current.front_face, current.right_face,
                current.left_face, current.back_face, current.down_face
            ] for row in face)
            
            if cube_state not in visited:
                visited.add(cube_state)

                for move in self.moves:
                    new_cube = current.copy()
                    new_cube.apply_move(move)
                    queue.append((new_cube, path + [move]))

        return None 
    
# dfs
class dfs:
    def __init__(self):
        self.cube = RubiksCube2x2()
        self.moves = ['R', 'L', 'U', 'D', 'F', 'B'] 
        self.solution = None # solution
        
    def solve(self):
        visited = set()  # visited
        
        def dfs(current, path):
            if current.is_solved():
                self.solution = path  
                return True
            
            cube_state = tuple(tuple(row) for face in [
                current.up_face, current.front_face, current.right_face,
                current.left_face, current.back_face, current.down_face
            ] for row in face)
            
            if cube_state in visited:
                return False
            
            visited.add(cube_state)
            
            for move in self.moves:
                new_cube = current.copy()
                new_cube.apply_move(move)
                if dfs(new_cube, path + [move]):
                    return True
            
            return False
        
        dfs(self.cube, [])  
        return self.solution
    
# ids

class ids:
    def __init__(self):
        self.cube = RubiksCube2x2()
        self.moves = ['R', 'L', 'U', 'D', 'F', 'B']
        self.solution = None
    
    def solve(self, max_depth=14):
        for depth in range(max_depth):
            visited = set()  # visited
            
            def dfs(current, path, current_depth):
                if current_depth == depth:
                    return False  # Reached depth limit
                
                if current.is_solved():
                    self.solution = path  
                    return True
                
                cube_state = tuple(tuple(row) for face in [
                    current.up_face, current.front_face, current.right_face,
                    current.left_face, current.back_face, current.down_face
                ] for row in face)
                
                if cube_state in visited:
                    return False
                
                visited.add(cube_state)
                
                for move in self.moves:
                    new_cube = current.copy()
                    new_cube.apply_move(move)
                    if dfs(new_cube, path + [move], current_depth + 1):
                        return True
                
                return False
            
            if dfs(self.cube, [], 0):
                return self.solution
        
        return None

    
# testing
cube = RubiksCube2x2()
cube.displayCube()

cube.apply_multMoves(['R', 'L']) 
# cube.apply_multMoves(['R', 'L', 'L', 'L', 'L']) 
# cube.apply_multMoves(['R', 'D', 'L']) 

cube.displayCube()

solver = bfs()
solver.cube = cube

start_time = time.time()
solution = solver.solve()
end_time = time.time()

print("Solution:", solution)
print("Time taken to solve:", round(end_time - start_time, 2), "seconds")

for move in solution:
    solver.cube.apply_move(move)
solver.cube.displayCube()