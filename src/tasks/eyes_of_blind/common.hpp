#pragma once
#include <array>
#include <cstdint>
namespace awakening::eyes_of_blind {
static constexpr std::size_t MAX_PACKET_SIZE = 300;

#pragma pack(push, 1)
struct PacketHeader {
    uint64_t sequence_id;
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 8, "Header must be 8 bytes");
static constexpr std::size_t HEADER_SIZE = sizeof(PacketHeader); // 8 bytes for header
static constexpr std::size_t PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE; // 292 bytes for payload

struct BlindSend {
    PacketHeader header;
    std::array<uint8_t, PAYLOAD_SIZE> data {};
};
} // namespace awakening::eyes_of_blind

// 注意：下位机收到的数据包经过protobuf序列化后，大小稍大于MAX_PACKET_SIZE
// 例如以下结构体：
// #pragma pack(push, 1)
// struct SerialPacket {
//     uint8_t  data[318];       // 固定缓冲区，容纳 Protobuf 序列化数据 + 预留
// };
// #pragma pack(pop)