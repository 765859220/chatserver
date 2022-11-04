#include "test.pb.h"
#include <iostream>
#include <string>
using namespace fixbug;

int main() {
    LoginRequest req;
    req.set_name("zhang san");
    req.set_pwd("123456");

    //定义存储序列化数据的字符串
    std::string send_str;

    //序列化
    if(req.SerializeToString(&send_str)) {
        std::cout << send_str << std::endl;
    }

    LoginRequest req2;
    if(req2.ParseFromString(send_str)) {
        std::cout << req2.name() << std::endl;
        std::cout << req2.pwd() << std::endl;
    }

    return 0;
}