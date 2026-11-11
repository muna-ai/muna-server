/*
*   Muna
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include <cpptrace/cpptrace.hpp>

namespace {

constexpr size_t MAX_FRAMES = 192;
constexpr int32_t REPORT_TIMEOUT_MS = 20000;
constexpr int32_t FATAL_SIGNALS[]{ SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE, SIGTRAP };
int32_t reporter_socket = -1;
std::atomic_flag reporting = ATOMIC_FLAG_INIT;

struct Report {
    int32_t pid;
    int32_t tid;
    int32_t signal;
    int32_t code;
    uintptr_t address;
    uintptr_t pc;
    uintptr_t sp;
    uint32_t count;
};

// Only the first crashing thread writes these buffers. Never allocate on a
// possibly exhausted or corrupted stack, or through a possibly locked malloc.
Report report{};
cpptrace::frame_ptr addresses[MAX_FRAMES]{};
cpptrace::safe_object_frame frames[MAX_FRAMES]{};

struct AlternateStack {
    void* memory = MAP_FAILED;
    size_t size = 0;
    ~AlternateStack() noexcept {
        if (memory == MAP_FAILED)
            return;
        stack_t disabled{};
        disabled.ss_flags = SS_DISABLE;
        if (sigaltstack(&disabled, nullptr) == 0)
            munmap(memory, size);
    }
};
thread_local AlternateStack alternate_stack;

// Used only with constant literals or our fixed-size number buffer. No stdio,
// formatting library, dynamic loader, or Rust runtime calls in the handler.
void write_bytes(
    const char* text,
    size_t size
) noexcept {
    while (size > 0) {
        const ssize_t written = write(STDERR_FILENO, text, size);
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0)
            return;
        text += written;
        size -= static_cast<size_t>(written);
    }
}

template<size_t N>
void literal(const char (&text)[N]) noexcept {
    write_bytes(text, N - 1);
}

void number(
    uintptr_t value,
    uint32_t base = 16
) noexcept {
    char buffer[32];
    size_t cursor = sizeof(buffer);
    do {
        buffer[--cursor] = "0123456789abcdef"[value % base];
        value /= base;
    } while (value != 0);
    write_bytes(buffer + cursor, sizeof(buffer) - cursor);
}

int64_t monotonic_ms() noexcept {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<int64_t>(now.tv_sec) * 1000 + now.tv_nsec / 1000000;
}

bool wait_socket(
    int16_t events,
    int64_t deadline
) noexcept {
    while (true) {
        const int64_t remaining = deadline - monotonic_ms();
        if (remaining <= 0)
            return false;
        pollfd fd{ .fd = reporter_socket, .events = events, .revents = 0 };
        const int32_t result = poll(&fd, 1, static_cast<int32_t>(remaining));
        if (result < 0 && errno == EINTR)
            continue;
        return result > 0 && (fd.revents & events) != 0;
    }
}

bool send_bytes(
    const void* data,
    size_t size,
    int64_t deadline
) noexcept {
    const auto* cursor = static_cast<const char*>(data);
    while (size > 0) {
        const ssize_t sent = send(reporter_socket, cursor, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR)
            continue;
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!wait_socket(POLLOUT, deadline))
                return false;
            continue;
        }
        if (sent <= 0)
            return false;
        cursor += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

bool read_bytes(
    int32_t fd,
    void* data,
    size_t size
) noexcept {
    auto* cursor = static_cast<char*>(data);
    while (size > 0) {
        const ssize_t received = read(fd, cursor, size);
        if (received < 0 && errno == EINTR)
            continue;
        if (received <= 0)
            return false;
        cursor += received;
        size -= static_cast<size_t>(received);
    }
    return true;
}

void fatal_signal(
    int32_t signal,
    siginfo_t* info,
    void* context
) noexcept {
    if (reporting.test_and_set(std::memory_order_relaxed)) {
        // Let the first reporter finish before process-wide termination.
        // All handled signals are masked on this thread until then.
        for (;;)
            pause();
    }
    report.pid = getpid();
    report.tid = static_cast<int32_t>(syscall(SYS_gettid));
    report.signal = signal;
    report.code = info ? info->si_code : 0;
    report.address = info ? reinterpret_cast<uintptr_t>(info->si_addr) : 0;
    auto* registers = static_cast<ucontext_t*>(context);
#if defined(__x86_64__)
    report.pc = static_cast<uintptr_t>(registers->uc_mcontext.gregs[REG_RIP]);
    report.sp = static_cast<uintptr_t>(registers->uc_mcontext.gregs[REG_RSP]);
#elif defined(__aarch64__)
    report.pc = registers->uc_mcontext.pc;
    report.sp = registers->uc_mcontext.sp;
#endif
    timespec time{};
    clock_gettime(CLOCK_REALTIME, &time);
    literal("\n=== MUNA FATAL SIGNAL ===\nunix_time_s="); number(time.tv_sec, 10);
    literal(" pid="); number(report.pid, 10);
    literal(" tid="); number(report.tid, 10);
    literal(" signal="); number(signal, 10);
    literal(" si_code=0x"); number(static_cast<uint32_t>(report.code));
    literal(" fault_address=0x"); number(report.address);
    literal(" pc=0x"); number(report.pc);
    literal(" sp=0x"); number(report.sp);
    literal("\n");

    // Write the saved fault PC and its module offset before attempting a walk:
    // even an unusable stack must leave something we can symbolicate offline.
    const auto log_frame = [](size_t i) noexcept {
        cpptrace::get_safe_object_frame(addresses[i], &frames[i]);
        frames[i].object_path[sizeof(frames[i].object_path) - 1] = '\0';
        literal("frame "); number(i, 10);
        literal(" pc=0x"); number(addresses[i]);
        literal(" module=");
        size_t length = 0;
        while (length < sizeof(frames[i].object_path) && frames[i].object_path[length] != '\0')
            ++length;
        write_bytes(frames[i].object_path, length);
        literal(" offset=0x"); number(frames[i].address_relative_to_object_start);
        literal("\n");
    };
    addresses[0] = report.pc;
    log_frame(0);
    report.count = 1 + static_cast<uint32_t>(cpptrace::safe_generate_raw_trace(
        addresses + 1,
        MAX_FRAMES - 1
    ));
    for (size_t i = 1; i < report.count; ++i)
        log_frame(i);
    const int64_t deadline = monotonic_ms() + REPORT_TIMEOUT_MS;
    const bool sent = (
        reporter_socket >= 0 &&
        send_bytes(&report, sizeof(report), deadline) &&
        send_bytes(frames, report.count * sizeof(frames[0]), deadline)
    );
    char done = 0;
    if (!sent || !wait_socket(POLLIN, deadline) || recv(reporter_socket, &done, 1, 0) != 1 || done != 'D')
        literal("MUNA: symbolication helper unavailable or timed out; raw frames above remain usable\n");
    if (report.count == MAX_FRAMES)
        literal("MUNA: stack reached the 192-frame capture limit\n");
    literal("=== END MUNA FATAL SIGNAL ===\n");

    // This is a host-process decision, never recovery inside a model library.
    // Restore the default disposition and re-deliver on the faulting thread,
    // preserving the original signal for the supervisor and core collector.
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigaction(signal, &action, nullptr);
    syscall(SYS_tgkill, report.pid, report.tid, signal);
}

void print_proc_file(
    int32_t pid,
    const char* file
) {
    const std::string path = "/proc/" + std::to_string(pid) + "/" + file;
    std::ifstream input(path);
    std::fprintf(stderr, "--- %s ---\n", path.c_str());
    std::string line;
    size_t lines = 0;
    while (std::getline(input, line)) {
        if (++lines > 4096) {
            std::fprintf(stderr, "[truncated at 4096 lines]\n");
            break;
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    }
}

} // namespace

extern "C" int32_t muna_crash_register_thread() noexcept {
    auto& stack = alternate_stack;
    if (stack.memory != MAP_FAILED)
        return 0;
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        return EINVAL;
    const size_t page = static_cast<size_t>(page_size);
    stack.size = 256 * 1024 + page;
    stack.memory = mmap(nullptr, stack.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (stack.memory == MAP_FAILED)
        return errno;
    if (mprotect(stack.memory, page, PROT_NONE) != 0) {
        const int32_t error = errno;
        munmap(stack.memory, stack.size);
        stack.memory = MAP_FAILED;
        return error;
    }
    stack_t config{};
    config.ss_sp = static_cast<char*>(stack.memory) + page;
    config.ss_size = stack.size - page;
    if (sigaltstack(&config, nullptr) != 0) {
        const int32_t error = errno;
        munmap(stack.memory, stack.size);
        stack.memory = MAP_FAILED;
        return error;
    }
    return 0;
}

extern "C" int32_t muna_crash_reporter(int32_t fd) noexcept {
    // The helper never installs signal handlers and never initializes Tokio.
    // Only it may allocate, resolve symbols, or use buffered output.
    try {
        const char ready = 'R';
        if (send(fd, &ready, 1, MSG_NOSIGNAL) != 1)
            return 1;
        Report incoming{};
        if (!read_bytes(fd, &incoming, sizeof(incoming)))
            return 0; // Normal parent exit closes the socket.
        if (incoming.count > MAX_FRAMES)
            return 2;
        cpptrace::object_trace trace;
        for (size_t i = 0; i < incoming.count; ++i) {
            cpptrace::safe_object_frame frame{};
            if (!read_bytes(fd, &frame, sizeof(frame)))
                return 2;
            trace.frames.push_back(frame.resolve());
        }
        std::fprintf(stderr, "--- MUNA symbolicated crashing-thread stack (fault PC first) ---\n");
        trace.resolve().print(std::cerr, false);
        std::cerr.flush();
        print_proc_file(incoming.pid, "status");
        print_proc_file(incoming.pid, "maps");
        std::fflush(stderr);
        const char done = 'D';
        send(fd, &done, 1, MSG_NOSIGNAL);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MUNA: symbolication helper failed: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "MUNA: symbolication helper failed with an unknown exception\n");
    }
    return 1;
}

extern "C" int32_t muna_crash_install() noexcept {
    // Must run before the host creates any threads. The child only performs
    // async-signal-safe operations between fork and exec.
    if (!cpptrace::can_signal_safe_unwind() || !cpptrace::can_get_safe_object_frame())
        return ENOTSUP;
    const size_t count = cpptrace::safe_generate_raw_trace(addresses, MAX_FRAMES);
    for (size_t i = 0; i < count; ++i)
        cpptrace::get_safe_object_frame(addresses[i], &frames[i]);
    if (const int32_t error = muna_crash_register_thread(); error != 0)
        return error;
    int32_t sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0)
        return errno;
    char descriptor[32];
    std::snprintf(descriptor, sizeof(descriptor), "%d", sockets[1]);
    const pid_t child = fork();
    if (child == 0) {
        close(sockets[0]);
        fcntl(sockets[1], F_SETFD, 0);
        execl("/proc/self/exe", "muna-server", "--muna-crash-reporter", descriptor, nullptr);
        _exit(127);
    }
    const int32_t fork_error = errno;
    close(sockets[1]);
    if (child < 0) {
        close(sockets[0]);
        return fork_error;
    }
    reporter_socket = sockets[0];
    fcntl(reporter_socket, F_SETFL, O_NONBLOCK);
    char ready = 0;
    if (!wait_socket(POLLIN, monotonic_ms() + 5000) || recv(reporter_socket, &ready, 1, 0) != 1 || ready != 'R') {
        close(reporter_socket);
        reporter_socket = -1;
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
        return EIO;
    }
    struct sigaction action{};
    action.sa_sigaction = fatal_signal;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigfillset(&action.sa_mask);
    for (const int32_t signal : FATAL_SIGNALS) {
        if (sigaction(signal, &action, nullptr) != 0)
            return errno;
    }
    std::fprintf(stderr, "MUNA: cpptrace fatal-signal reporting enabled (helper_pid=%d, max_frames=%zu)\n", child, MAX_FRAMES);
    return 0;
}
