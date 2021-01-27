def gra():
    board = [1,2,3,4,5,6,7,8,9]
    
    choice1 = input("Player 1 - X or Y ?: ")
        
    while choice1 not in ['X', 'x', 'Y', 'y']:
        choice1 = input("Player 1 - X or Y ?: ")
    
    if choice1 in ['X','x']:
        choice1 = 'X'
        choice2 = 'Y'
    else:
        choice1 = 'Y'
        choice2 = 'X'
            
    for i in range(9):
        if board[6] == board[7] == board[8]:
            print('KONIEC')
            break
        elif board[3] == board[4] == board[5]:
            print('KONIEC')
            break
        elif board[0] == board[1] == board[2]:
            print('KONIEC')
            break
        elif board[6] == board[3] == board[0]:
            print('KONIEC')
            break
        elif board[7] == board[4] == board[1]:
            print('KONIEC')
            break
        elif board[8] == board[5] == board[2]:
            print('KONIEC')
            break
        elif board[6] == board[4] == board[2]:
            print('KONIEC')
            break
        elif board[0] == board[4] == board[8]:
            print('KONIEC')
            break
        else:
            result1 = int(input("P1, Please enter a number: "))
            
            
            board[result1 - 1]  = choice1 
            
            print(board[6],board[7],board[8])
            print(board[3],board[4],board[5])
            print(board[0],board[1],board[2])
        
            result2 = int(input("P2, Please enter a number: "))


            board[result2 - 1]  = choice2 
            
            print(board[6],board[7],board[8])
            print(board[3],board[4],board[5])
            print(board[0],board[1],board[2])