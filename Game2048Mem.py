import random


# Copied this from https://projectgurukul.org/python-2048-game/


class Board:
    def __init__(self):
        self.n = 4
        self.board = []
        self.gridCell = [[0] * 4 for i in range(4)]
        self.compress = False
        self.merge = False
        self.moved = False
        self.score = 0

    def reverse(self):
        for ind in range(4):
            i = 0
            j = 3
            while i < j:
                self.gridCell[ind][i], self.gridCell[ind][j] = self.gridCell[ind][j], self.gridCell[ind][i]
                i += 1
                j -= 1

    def transpose(self):
        self.gridCell = [list(t) for t in zip(*self.gridCell)]

    def compress_grid(self):
        self.compress = False
        temp = [[0] * 4 for i in range(4)]
        for i in range(4):
            cnt = 0
            for j in range(4):
                if self.gridCell[i][j] != 0:
                    temp[i][cnt] = self.gridCell[i][j]
                    if cnt != j:
                        self.compress = True
                    cnt += 1
        self.gridCell = temp

    def merge_grid(self):
        self.merge = False
        for i in range(4):
            for j in range(4 - 1):
                if self.gridCell[i][j] == self.gridCell[i][j + 1] and self.gridCell[i][j] != 0:
                    self.gridCell[i][j] *= 2
                    self.gridCell[i][j + 1] = 0
                    self.score += self.gridCell[i][j]
                    self.merge = True

    def can_merge(self):
        for i in range(4):
            for j in range(3):
                if self.gridCell[i][j] == self.gridCell[i][j + 1]:
                    return True

        for i in range(3):
            for j in range(4):
                if self.gridCell[i + 1][j] == self.gridCell[i][j]:
                    return True
        return False

    def random_cell(self):
        cells = []
        for i in range(4):
            for j in range(4):
                if self.gridCell[i][j] == 0:
                    cells.append((i, j))
        curr = random.choice(cells)
        i = curr[0]
        j = curr[1]
        if random.random() < 0.9:
            self.gridCell[i][j] = 2
        else:
            self.gridCell[i][j] = 4


class Game2048Mem:
    game_panel: Board

    def __init__(self, gamepanel):
        self.game_panel = gamepanel
        self.end = False
        self.won = False

    def start(self):
        self.game_panel.random_cell()
        self.game_panel.random_cell()

    def link_keys(self, direction):
        if self.end or self.won:
            return
        self.game_panel.compress = False
        self.game_panel.merge = False
        self.game_panel.moved = False
        if direction == 0:
            self.game_panel.transpose()
            self.game_panel.compress_grid()
            self.game_panel.merge_grid()
            self.game_panel.moved = self.game_panel.compress or self.game_panel.merge
            self.game_panel.compress_grid()
            self.game_panel.transpose()
        elif direction == 2:
            self.game_panel.transpose()
            self.game_panel.reverse()
            self.game_panel.compress_grid()
            self.game_panel.merge_grid()
            self.game_panel.moved = self.game_panel.compress or self.game_panel.merge
            self.game_panel.compress_grid()
            self.game_panel.reverse()
            self.game_panel.transpose()
        elif direction == 3:
            self.game_panel.compress_grid()
            self.game_panel.merge_grid()
            self.game_panel.moved = self.game_panel.compress or self.game_panel.merge
            self.game_panel.compress_grid()
        elif direction == 1:
            self.game_panel.reverse()
            self.game_panel.compress_grid()
            self.game_panel.merge_grid()
            self.game_panel.moved = self.game_panel.compress or self.game_panel.merge
            self.game_panel.compress_grid()
            self.game_panel.reverse()
        else:
            pass

        flag = 0
        for i in range(4):
            for j in range(4):
                if self.game_panel.gridCell[i][j] == 2048:
                    flag = 1
                    break
        if flag == 1:  # found 2048
            self.won = True
            print("won")
            return
        for i in range(4):
            for j in range(4):
                if self.game_panel.gridCell[i][j] == 0:
                    flag = 1
                    break
        if not (flag or self.game_panel.can_merge()):
            self.end = True
            print("Over")
        if self.game_panel.moved:
            self.game_panel.random_cell()
