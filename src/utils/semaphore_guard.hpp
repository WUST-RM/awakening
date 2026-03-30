#include <semaphore>
namespace awakening::utils {

class SemaphoreGuard {
public:
    explicit SemaphoreGuard(std::counting_semaphore<>& sem, bool acquired):
        sem_(sem),
        acquired_(acquired) {}
    ~SemaphoreGuard() {
        if (acquired_) {
            sem_.release();
        }
    }

private:
    std::counting_semaphore<>& sem_;
    bool acquired_;
};
} // namespace awakening::utils
