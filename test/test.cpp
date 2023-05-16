#include <vector>
#include <iostream>


int main(){
  std::vector<char> charVector = {1, 2, 3, 4};
  // auto num_vals = charVector.size();
  // int *p = reinterpret_cast<int*>(&(charVector)[0]);
  std::vector<int> intVector(charVector.begin(), charVector.end());

  // Print the contents of intVector
  for (int num : intVector) {
      std::cout << num << " ";
  }
  std::cout << std::endl;

  return 0;  
}