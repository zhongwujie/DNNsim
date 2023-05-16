#include"base/cnpy.h"
#include<complex>
#include<cstdlib>
#include<algorithm>
#include<cstring>
#include<iomanip>
#include<stdint.h>
#include<stdexcept>
#include <regex>

namespace base {

    char BigEndianTest() {
        int x = 1;
        return (((char *) &x)[0]) ? '<' : '>';
    }

    char map_type(const std::type_info &t) {
        if (t == typeid(float)) return 'f';
        if (t == typeid(double)) return 'f';
        if (t == typeid(long double)) return 'f';

        if (t == typeid(int)) return 'i';
        if (t == typeid(char)) return 'i';
        if (t == typeid(short)) return 'i';
        if (t == typeid(long)) return 'i';
        if (t == typeid(long long)) return 'i';

        if (t == typeid(unsigned char)) return 'u';
        if (t == typeid(unsigned short)) return 'u';
        if (t == typeid(unsigned long)) return 'u';
        if (t == typeid(unsigned long long)) return 'u';
        if (t == typeid(unsigned int)) return 'u';

        if (t == typeid(bool)) return 'b';

        if (t == typeid(std::complex<float>)) return 'c';
        if (t == typeid(std::complex<double>)) return 'c';
        if (t == typeid(std::complex<long double>)) return 'c';

        else return '?';
    }

    template<>
    std::vector<char> &operator+=(std::vector<char> &lhs, const std::string rhs) {
        lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        return lhs;
    }

    template<>
    std::vector<char> &operator+=(std::vector<char> &lhs, const char *rhs) {
        //write in little endian
        size_t len = strlen(rhs);
        lhs.reserve(len);
        for (size_t byte = 0; byte < len; byte++) {
            lhs.push_back(rhs[byte]);
        }
        return lhs;
    }

    void parse_npy_header(FILE *fp, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order) {
        char buffer[256];
        size_t res = fread(buffer, sizeof(char), 11, fp);
        if (res != 11)
            throw std::runtime_error("parse_npy_header: failed fread");
        std::string header = fgets(buffer, 256, fp);
        assert(header[header.size() - 1] == '\n');

        size_t loc1, loc2;

        //fortran order
        loc1 = header.find("fortran_order");
        if (loc1 == std::string::npos)
            throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
        loc1 += 16;
        fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

        //shape
        loc1 = header.find("(");
        loc2 = header.find(")");
        if (loc1 == std::string::npos || loc2 == std::string::npos)
            throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

        std::regex num_regex("[0-9][0-9]*");
        std::smatch sm;
        shape.clear();

        std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
        while (std::regex_search(str_shape, sm, num_regex)) {
            shape.push_back(std::stoi(sm[0].str()));
            str_shape = sm.suffix().str();
        }

        //endian, word size, data type
        //byte order code | stands for not applicable.
        //not sure when this applies except for byte array
        loc1 = header.find("descr");
        if (loc1 == std::string::npos)
            throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
        loc1 += 9;
        bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
        assert(littleEndian);

        //char type = header[loc1+1];
        //assert(type == map_type(T));

        std::string str_ws = header.substr(loc1 + 2);
        loc2 = str_ws.find("'");
        word_size = atoi(str_ws.substr(0, loc2).c_str());
    }


    NpyArray load_the_npy_file(FILE *fp, std::vector<size_t> &shape) {
        size_t word_size;
        bool fortran_order;
        base::parse_npy_header(fp, word_size, shape, fortran_order);
        base::NpyArray arr(shape, word_size, fortran_order);
        size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
        if (nread != arr.num_bytes())
            throw std::runtime_error("load_the_npy_file: failed fread");
        return arr;
    }

    void npy_load(std::string fname, NpyArray &array, std::vector<size_t> &shape) {
        FILE *fp = fopen(fname.c_str(), "rb");
        if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);
        array = load_the_npy_file(fp, shape);
        fclose(fp);
    }

}


