#ifndef ACTIVATORS_H
#define ACTIVATORS_H

inline auto sigmoid(f32 x) -> f32 {
    return 1.f / (1.f + exp(-x));
}

#endif //ACTIVATORS_H
