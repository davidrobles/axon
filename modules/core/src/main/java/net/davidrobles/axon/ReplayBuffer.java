package net.davidrobles.axon;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * A fixed-capacity circular replay buffer for storing and sampling environment transitions.
 *
 * <p>New transitions are added in O(1). When the buffer is full, the oldest transition is
 * overwritten. Random sampling without replacement is supported for mini-batch updates.
 *
 * @param <S> the state type
 * @param <A> the action type
 */
public class ReplayBuffer<S, A> {
    private final int capacity;
    private final List<Transition<S, A>> buffer;
    private int writeIndex = 0;
    private int size = 0;

    /**
     * @param capacity maximum number of transitions to store; must be >= 1
     */
    public ReplayBuffer(int capacity) {
        if (capacity < 1)
            throw new IllegalArgumentException("capacity must be >= 1, got: " + capacity);
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
    }

    /**
     * Adds a transition to the buffer. If the buffer is full, the oldest transition is overwritten.
     *
     * @param transition the transition to store
     */
    public void add(Transition<S, A> transition) {
        Objects.requireNonNull(transition, "transition must not be null");
        if (size < capacity) {
            buffer.add(transition);
            size++;
        } else {
            buffer.set(writeIndex, transition);
        }
        writeIndex = (writeIndex + 1) % capacity;
    }

    /**
     * Returns a random sample of {@code batchSize} transitions without replacement.
     *
     * @param batchSize number of transitions to sample; must be <= {@link #size()}
     * @param rng random number generator
     * @return an unordered list of sampled transitions
     * @throws IllegalArgumentException if {@code batchSize} exceeds the current buffer size
     */
    public List<Transition<S, A>> sample(int batchSize, Random rng) {
        if (batchSize > size)
            throw new IllegalArgumentException(
                    "batchSize (" + batchSize + ") exceeds buffer size (" + size + ")");
        // Fisher-Yates partial shuffle to select batchSize indices without replacement
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) indices.add(i);
        List<Transition<S, A>> batch = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            int j = i + rng.nextInt(size - i);
            int tmp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, tmp);
            batch.add(buffer.get(indices.get(i)));
        }
        return batch;
    }

    /** Returns the number of transitions currently stored. */
    public int size() {
        return size;
    }

    /** Returns {@code true} if the buffer has reached its capacity. */
    public boolean isFull() {
        return size == capacity;
    }
}
