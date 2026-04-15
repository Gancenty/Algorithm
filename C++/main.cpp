#include <iostream>
#include <vector>

#include <Eigen/Eigen>

using namespace std;

int main() {
    auto a = std::numeric_limits<double>::infinity();
    auto b = a/2;
    if (a > b){
        cout << "a > b" << endl;
    }else if (a < b){
        cout << "a < b" << endl;
    }else{
        cout << "a == b" << endl;
    }
    return 0;
}