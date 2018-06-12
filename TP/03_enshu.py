def collatz(number):
    if number % 2 == 0:
        return number // 2
    elif number % 2 == 1:
        return 3 * number + 1
    else:
        print("Error")

try:
   print("整数を入力して下さい。")
   col = int(input())
   if col == 0:
       print("整数を入力して下さい。")
   else:
       while col != 1:
           col = collatz(col)
           print(col)

except ValueError as e:
    print("エラー：不正な引数です。整数を入力してください。")
