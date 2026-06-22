import cv2
import numpy as np

# Load two satellite images (side-by-side)
image_path = r'C:/Users/Student/Desktop/CV Images/Two satellite images of the same geographical region.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Error loading image. Check path.")
    exit()

h, w = image.shape[:2]
img1 = image[:, :w//2]  # Left: reference
img2 = image[:, w//2:]  # Right: source with shift/rotation/distortion

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB detector
orb = cv2.ORB_create(nfeatures=5000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract points (good matches)
src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)

# Estimate affine matrix
matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)
if matrix is None:
    print("Affine estimation failed. Using identity.")
    matrix = np.eye(2, 3, dtype=np.float32)


# Print derived affine matrix
print("Derived Affine Transformation Matrix:")
print(matrix)

# Warp source to reference
h1, w1 = img1.shape[:2]
aligned = cv2.warpAffine(img2, matrix, (w1, h1))

# Blend for visualization
blend = cv2.addWeighted(img1, 0.5, aligned, 0.5, 0)

# Stack horizontally for single window output
combined = np.hstack((img1, img2, aligned, blend))

cv2.imshow('Single Window: Ref | Source | Aligned | Blend', combined)
print("Derived Affine Matrix:")
print(matrix)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()





import numpy as np

GRID = 4
ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]  # R L D U

obstacles = [(1,1)]
penalty = {(2,2): -5}
goal = (3,3)


def step(state, action):
    if state == goal:
        return state, 0

    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
        return state, -1

    if (nr, nc) in obstacles:
        return state, -1

    if (nr, nc) in penalty:
        return (nr, nc), penalty[(nr, nc)]

    if (nr, nc) == goal:
        return (nr, nc), 10

    return (nr, nc), -1


def mc_control(epsilon=0.1, episodes=500):
    Q = {(r,c): np.zeros(4) for r in range(GRID) for c in range(GRID)}
    returns = {(r,c): [[] for _ in range(4)] for r in range(GRID) for c in range(GRID)}
    policy = {(r,c): np.ones(4)/4 for r in range(GRID) for c in range(GRID)}

    for _ in range(episodes):
        episode = []
        state = (0,0)

        while state != goal:
            action = np.random.choice(4, p=policy[state])
            next_state, reward = step(state, action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G += r

            if (s,a) not in visited:
                returns[s][a].append(G)
                Q[s][a] = np.mean(returns[s][a])
                visited.add((s,a))

                best_a = np.argmax(Q[s])
                for i in range(4):
                    if i == best_a:
                        policy[s][i] = 1 - epsilon + epsilon/4
                    else:
                        policy[s][i] = epsilon/4

    return policy


def show_policy(policy):
    arrows = ['→','←','↓','↑']
    print("\n=== Optimal Policy ===")
    for r in range(GRID):
        row = ""
        for c in range(GRID):
            if (r,c) in obstacles:
                row += " X "
            elif (r,c) == goal:
                row += " G "
            else:
                row += " " + arrows[np.argmax(policy[(r,c)])] + " "
        print(row)


policy = mc_control()
show_policy(policy)