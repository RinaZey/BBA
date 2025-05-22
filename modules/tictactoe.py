# modules/tictactoe.py

import math
from typing import List, Tuple

class TicTacToe:
    def __init__(self):
        # пустая 3×3 доска
        self.board: List[List[str]] = [[' ']*3 for _ in range(3)]
        self.player = 'X'
        self.bot    = 'O'

    def render(self) -> str:
        header = "  1 2 3"
        rows = []
        for i, row in enumerate(self.board):
            rows.append(
                chr(ord('A')+i) + ' ' +
                ' '.join(c if c!=' ' else '.' for c in row)
            )
        return header + '\n' + '\n'.join(rows)

    def is_full(self) -> bool:
        return all(cell!=' ' for row in self.board for cell in row)

    def check_win(self, symbol: str) -> bool:
        b = self.board
        lines = [
            [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
            [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
            [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)]
        ]
        return any(all(b[r][c]==symbol for r,c in line) for line in lines)

    def available_moves(self) -> List[Tuple[int,int]]:
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i,j))
        return moves

    def minimax(self, is_bot_turn: bool) -> Tuple[int, Tuple[int,int]]:
        """
        Возвращает (оценка, ход) по алгоритму minimax.
        Оценка: +1 — победа бота, -1 — победа игрока, 0 — ничья.
        """
        # проверяем терминальные состояния
        if self.check_win(self.bot):
            return  1, (-1,-1)
        if self.check_win(self.player):
            return -1, (-1,-1)
        if self.is_full():
            return  0, (-1,-1)

        best_move = (-1,-1)
        if is_bot_turn:
            best_score = -math.inf
            for (r,c) in self.available_moves():
                self.board[r][c] = self.bot
                score, _ = self.minimax(False)
                self.board[r][c] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (r,c)
            return best_score, best_move
        else:
            best_score = math.inf
            for (r,c) in self.available_moves():
                self.board[r][c] = self.player
                score, _ = self.minimax(True)
                self.board[r][c] = ' '
                if score < best_score:
                    best_score = score
                    best_move = (r,c)
            return best_score, best_move

    def bot_move(self) -> None:
        """Делает ход ботом по Minimax."""
        _, (r,c) = self.minimax(True)
        if (r,c) != (-1,-1):
            self.board[r][c] = self.bot

    def player_move(self, coord: str) -> Tuple[str,bool]:
        """
        Игрок делает ход. coord в формате "A1".."C3".
        Возвращает (сообщение, завершена_ли_игра).
        """
        coord = coord.strip().upper()
        if len(coord)!=2 or coord[0] not in "ABC" or coord[1] not in "123":
            return "Неверный формат! Вводи букву A–C и цифру 1–3, например A1.", False

        r = ord(coord[0]) - ord('A')
        c = int(coord[1]) - 1
        if self.board[r][c] != ' ':
            return "Эта клетка уже занята, выбери другую.", False

        # ход игрока
        self.board[r][c] = self.player
        # проверяем выигрыш
        if self.check_win(self.player):
            return self.render() + "\nПоздравляю, ты выиграл! 🎉", True

        # ничья?
        if self.is_full():
            return self.render() + "\nНичья! 🤝", True

        # ход бота
        self.bot_move()
        # проверяем выигрышь бота
        if self.check_win(self.bot):
            return self.render() + "\nУвы, бот выиграл 😢", True

        # ничья после хода бота?
        if self.is_full():
            return self.render() + "\nНичья! 🤝", True

        # продолжаем игру
        return self.render() + "\nТвой ход (формат A1..C3):", False
