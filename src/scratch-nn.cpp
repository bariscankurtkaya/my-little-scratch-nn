#include <iostream>
#define X 4
#define Y 5

void print_matrix(int matrix[][X]);

int main() {
    int matrix[Y][X] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}};
    print_matrix(matrix);
    return 0;
}

void print_matrix(int matrix[][X]){
    for(int i=0; i< Y; i++){
        for(int j=0; j< X; j++){
            printf("%d -", matrix[i][j]);
        }
        printf("\n");
    }
}
