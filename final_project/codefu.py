class SupperBowlWinner:
      def winner(_self_,gameLog):
        arr = []
        team1 = 0
        team2 = 0
        for ch in gameLog:
            arr = ch
        print(arr)
        for i in range(len(arr)-1):
            if is_number(arr[i+1]):
                if arr[i+1] == 1:
                    if arr[i] == 'T':
                        team1 +=7
                        i+=1
                        continue
                    elif arr[i] == 't':
                        team2+=7
                        i+=1
                        continue
                    elif arr[i+1] == 2:
                        if arr[i] == 'T':
                            team1 +=8
                            i+=1
                            continue
                        elif arr[i] == 't':
                            team2+=8
                            i+=1
                            continue
            elif arr[i] == 'T':
                team1+=6
                continue
            elif arr[i] == 't':
                team2+=6
                continue
            elif arr[i] == 'F':
                team1+=3
                continue
            elif arr[i] == 'f':
                team2+=3
                continue
            elif arr[i] == 'S':
                team1+=2
                continue
            elif arr[i] == 's':
                team2+=2
                continue
        if arr[len(arr)-1] == 'T':
            team1+=6
        elif arr[len(arr)-1] == 't':
            team2+=6
        elif arr[len(arr)-1] == 'F':
            team1+=3
        elif arr[len(arr)-1] == 'f':
            team2+=3
        elif arr[len(arr)-1] == 'S':
            team1+=2
        elif arr[len(arr)-1] == 's':
            team2+=2
        if team1 > team2:
            return "Patriots won {} : {}".format(team1,team2)
        elif team1 < team2:
            return "Eagles won {} : {}".format(team1,team2)
        else:
            return "Draw {} : {}".format(team1,team2) 