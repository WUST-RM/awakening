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

static constexpr std::size_t HEADER_SIZE = sizeof(PacketHeader);
static constexpr std::size_t PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE;

struct BlindSend {
    PacketHeader header;
    std::array<uint8_t, PAYLOAD_SIZE> data {};
};
} // namespace awakening::eyes_of_blind