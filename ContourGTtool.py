import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting 지원
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil

# === [경로 설정] ===
SAVE_DIR = '/home/ohj/contour_GT/'
os.makedirs(SAVE_DIR, exist_ok=True)


GROUND_MASK_PATH = '/home/ohj/ground_mask.npy'

K = np.array([[487.528145896324, 0, 304.18549882534194],
              [0, 487.9677945726932, 181.52548268646],
              [0, 0, 1]], dtype=np.float32)

# K = np.array([[975.056291792648, 0, 608.3709976506839],
#               [0, 975.9355891453864, 363.05096537292],
#               [0, 0, 1]], dtype=np.float32)
import matplotlib.pyplot as plt
import numpy as np

def generate_ground_mask(IMAGE_PATH, SAVE_PATH):
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(param['img'], (x, y), 3, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(param['img'], points[-2], points[-1], (255, 0, 0), 2)
            cv2.imshow('Click Polygon Points', param['img'])

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] 이미지 경로 확인 필요: {IMAGE_PATH}")
        return

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    display_img = img.copy()

    cv2.imshow('Click Polygon Points', display_img)
    cv2.setMouseCallback('Click Polygon Points', on_click, param={'img': display_img})

    print("👆 왼쪽 클릭: Polygon 점 선택")
    print("⏎ Enter: 다각형 닫고 마스크 생성 및 저장")
    print("⎋ ESC: 종료")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.array(points)], 1)
                np.save(SAVE_PATH, mask)
                print(f"[✔] Ground mask saved to {SAVE_PATH}")
            else:
                print("[!] 최소 3개의 점이 필요합니다.")
            break

        elif key == 27:  # ESC
            print("[x] 종료됨")
            break

    cv2.destroyAllWindows()

def interactive_scatter_removal(points, xlim=(-16, 16), ylim=(0, 32), title="Drag to remove points", height_image_path=None):
    """
    points: (N, 2) ndarray of (x, z) coordinates
    height_image_path: height_layer_Xm.png 경로
    """
    points = np.array(points)
    selected_rect = []
    rect_patch = [None]  # 리스트로 만들어야 내부 함수에서 갱신 가능

    # 오른쪽에 띄울 이미지 (OpenCV -> RGB)
    height_img = None
    if height_image_path and os.path.exists(height_image_path):
        height_img = cv2.cvtColor(cv2.imread(height_image_path), cv2.COLOR_BGR2RGB)

    def on_press(event):
        if event.inaxes:
            selected_rect.clear()
            selected_rect.append((event.xdata, event.ydata))

            if rect_patch[0]:
                rect_patch[0].remove()
                rect_patch[0] = None

    def on_motion(event):
        if event.inaxes and len(selected_rect) == 1:
            x0, y0 = selected_rect[0]
            x1, y1 = event.xdata, event.ydata
            width, height = x1 - x0, y1 - y0

            if rect_patch[0]:
                rect_patch[0].remove()

            rect_patch[0] = plt.Rectangle((x0, y0), width, height,
                                          linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')
            ax_left.add_patch(rect_patch[0])
            fig.canvas.draw_idle()

    def on_release(event):
        if event.inaxes and len(selected_rect) == 1:
            selected_rect.append((event.xdata, event.ydata))
            x0, y0 = selected_rect[0]
            x1, y1 = selected_rect[1]

            nonlocal points
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            mask = (points[:, 0] < x_min) | (points[:, 0] > x_max) | (points[:, 1] < y_min) | (points[:, 1] > y_max)
            points = points[mask]

            if rect_patch[0]:
                rect_patch[0].remove()
                rect_patch[0] = None

            ax_left.clear()
            ax_left.scatter(points[:, 0], points[:, 1], s=5)
            ax_left.set_xlim(*xlim)
            ax_left.set_ylim(*ylim)
            ax_left.set_title(title)
            fig.canvas.draw_idle()

    # === Subplot 기반 시각화 ===
    if height_img is not None:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7))
    else:
        fig, ax_left = plt.subplots(figsize=(10, 10))
        ax_right = None

    ax_left.scatter(points[:, 0], points[:, 1], s=5)
    ax_left.set_xlim(*xlim)
    ax_left.set_ylim(*ylim)
    ax_left.set_title(title)

    if ax_right is not None:
        ax_right.imshow(height_img)
        ax_right.set_title("RGB Height Layer")
        ax_right.axis('off')

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.tight_layout()
    plt.show()
    return points

    
def pixel_to_3d(x, y, depth, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.array([X, Y, Z])

def fit_ground_plane_ransac(points_3d):
    X = points_3d[:, :2]
    Z = points_3d[:, 2]
    ransac = RANSACRegressor()
    ransac.fit(X, Z)
    coef = ransac.estimator_.coef_
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    point_on_plane = np.mean(points_3d, axis=0)
    return normal, point_on_plane

def project_to_image(P, K):
    if P[2] == 0 or not np.isfinite(P[2]):
        return None
    x = (K[0, 0] * P[0] / P[2]) + K[0, 2]
    y = (K[1, 1] * P[1] / P[2]) + K[1, 2]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return int(round(x)), int(round(y))
    
def load_height_layer_image(h_val):
    file_path = f"height_layer_{h_val}m.png"
    if os.path.exists(file_path):
        return cv2.imread(file_path)
    else:
        print(f"[WARN] {file_path} 파일 없음")
        return None
def visualize_3m_plane_points(mode="stereo"):
    prefix = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    layer_save_dir = os.path.join(SAVE_DIR, 'contours', prefix)
    os.makedirs(layer_save_dir, exist_ok=True)

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map_raw = np.load(DEPTH_PATH)

    if mode == "stereo":
        depth_map = np.nan_to_num(depth_map_raw, nan=0.0, posinf=0.0, neginf=0.0)
        depth_map[depth_map <= 0] = 0.0
    elif mode == "estimated":
        depth_map = depth_map_raw.copy()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    ground_mask = np.load(GROUND_MASK_PATH).astype(bool)
    h, w = depth_map.shape
    ys, xs = np.where(ground_mask)
    depths_ground = depth_map[ys, xs]
    valid = depths_ground > 0
    ground_points = np.array([pixel_to_3d(x, y, d, K) for x, y, d in zip(xs[valid], ys[valid], depths_ground[valid])])

    normal, p0 = fit_ground_plane_ransac(ground_points)

    all_pixels = np.indices((h, w)).transpose(1, 2, 0).reshape(-1, 2)
    all_depths = depth_map.flatten()
    pseudo_lidar = np.array([
        pixel_to_3d(x, y, z, K)
        for (y, x), z in zip(all_pixels, all_depths) if z > 0
    ])
    pseudo_lidar = pseudo_lidar[pseudo_lidar[:, 2] <= 25.0]

    display_img1 = image.copy()
    full_point_set = set()
    point_to_3d_map = {}

    for h_val in range(1, 6):
    # 평면 위의 한 점은 p0 + h_val * normal
    # 평면에서의 법선 거리 계산
        dists = np.abs(np.dot((pseudo_lidar - (p0 + h_val * normal)), normal))

        pixel_coords = [project_to_image(p, K) for p in pseudo_lidar]

        selected_pts = {}
        for i, pix in enumerate(pixel_coords):
            if pix is None:
                continue
            u, v = pix
            if u not in selected_pts or dists[i] < selected_pts[u][0]:
                selected_pts[u] = (dists[i], pix, pseudo_lidar[i])

        color = color_per_h[h_val] if 'color_per_h' in locals() else (255 - h_val * 20, 20 * h_val, 50 + 15 * h_val)
        img = display_img_per_height[h_val] if 'display_img_per_height' in locals() else display_img1

        for _, (dist, pix, point3d) in selected_pts.items():
            full_point_set.add(pix)
            point_to_3d_map[pix] = (h_val, point3d)
            cv2.circle(display_img1, pix, 1, color, -1)  # 종합 이미지
            cv2.circle(img, pix, 1, color, -1)           # 개별 이미지


    dragging = False
    start_pt = (0, 0)

    def draw_rectangle(img, pt1, pt2):
        temp = img.copy()
        cv2.rectangle(temp, pt1, pt2, (0, 255, 0), 1)
        return temp

    def erase_within_rect(x1, y1, x2, y2):
        nonlocal full_point_set
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        full_point_set = {pt for pt in full_point_set if not (x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max)}

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, start_pt, display_img1

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            temp_img = draw_rectangle(display_img1, start_pt, (x, y))
            cv2.imshow("Editable Height Lines", np.hstack((temp_img, image)))
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            erase_within_rect(start_pt[0], start_pt[1], x, y)
            display_img1 = image.copy()
            for pt in full_point_set:
                cv2.circle(display_img1, pt, 1, (0, 0, 255), -1)
            cv2.imshow("Editable Height Lines", np.hstack((display_img1, image)))

    for pt in full_point_set:
        cv2.circle(display_img1, pt, 1, (0, 0, 255), -1)

    cv2.imshow("Editable Height Lines", display_img1)
    cv2.setMouseCallback("Editable Height Lines", mouse_callback)

    print("🖱 마우스로 드래그: 1~10m 라인 점 지우기 | ⏎ Enter: 확정 종료 | ⎋ ESC: 취소")
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        elif k == 27:
            print("작업 취소됨")
            return

    cv2.destroyAllWindows()

    refined_points_by_height = {i: [] for i in range(1, 6)}
    for pt in full_point_set:
        if pt in point_to_3d_map:
            h_val, point3d = point_to_3d_map[pt]
            refined_points_by_height[h_val].append(point3d)

    display_img_per_height = {i: image.copy() for i in range(1, 6)}
    color_per_h = {h_val: (255 - h_val * 20, 20 * h_val, 50 + 15 * h_val) for h_val in range(1, 6)}

    for h_val in range(1, 6):
        plane_point_h = p0 + h_val * normal
        diff = pseudo_lidar - plane_point_h
        dists = np.abs(np.dot(diff, normal))

        pixel_coords = [project_to_image(p, K) for p in pseudo_lidar]

        selected_pts = {}
        for i, pix in enumerate(pixel_coords):
            if pix is None:
                continue
            u, v = pix
            if u not in selected_pts or dists[i] < selected_pts[u][0]:
                selected_pts[u] = (dists[i], pix, pseudo_lidar[i])

        color = color_per_h[h_val]
        img = display_img_per_height[h_val]

        for _, (dist, pix, point3d) in selected_pts.items():
            full_point_set.add(pix)
            point_to_3d_map[pix] = (h_val, point3d)
            cv2.circle(display_img1, pix, 1, color, -1)  # 종합 이미지
            cv2.circle(img, pix, 3, color, -1)           # 개별 높이 이미지

    for h_val in range(1, 6):
        pts = refined_points_by_height.get(h_val, [])
        print(f"[DEBUG] h={h_val}m: {len(pts)} points")

        valid_pix = 0
        for pt in pts:
            pix = project_to_image(pt, K)
            if pix is not None:
                valid_pix += 1
        print(f"[DEBUG] h={h_val}m: {valid_pix} valid projected points")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # === z축 별 시각화 이미지 저장 ===
    if 1:
        for h_val in range(1, 6):
            img = image.copy()
            color = color_per_h[h_val]
            pts = refined_points_by_height[h_val]

            print(f"[DEBUG] {h_val}m layer: {len(pts)} points")

            has_valid = False
            for pt in pts:
                pix = project_to_image(pt, K)
                if pix is not None:
                    has_valid = True
                    cv2.circle(img, pix, 1, color, -1)

            if has_valid:
                save_path = os.path.join("/home/ohj", f"height_layer_{h_val}m.png")
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"[INFO] Saved {save_path}")
            else:
                print(f"[WARN] {h_val}m layer has no valid projectable points — skipped saving.")

    # === Bird's Eye View 시각화 ===
    plt.figure(figsize=(10, 10))
    cmap = cm.get_cmap('tab10', 10)
    for h_val in range(1, 6):
        pts = np.array(refined_points_by_height[h_val])
        bev_pts = pts[:, [0, 2]]
        height_img_path = f"height_layer_{h_val}m.png"

        filtered_bev_pts = interactive_scatter_removal(
            bev_pts,
            xlim=(-16, 16),
            ylim=(0, 32),
            title=f"Drag to remove - {h_val}m",
            height_image_path=height_img_path
        )
        # 필터링된 점들을 다시 3D로 복원 (X는 고정, Z는 고정, Y는 기존에서 추정)
        filtered_3d_pts = []
        for x, z in filtered_bev_pts:
            matches = pts[(np.abs(pts[:, 0] - x) < 1e-3) & (np.abs(pts[:, 2] - z) < 1e-3)]
            if len(matches) > 0:
                filtered_3d_pts.append(matches[0])

        refined_points_by_height[h_val] = np.array(filtered_3d_pts)
    #     if len(pts) == 0:
    #         continue
    #     xs = pts[:, 0]
    #     ys = pts[:, 2]  # Z축이 높이, X-Z 평면 투영
    #     plt.scatter(xs, ys, color=cmap(h_val - 1), s=4, label=f'{h_val}m')

    # plt.xlabel('X axis [m]')
    # plt.ylabel('Z axis [m] (depth forward)')
    # plt.title("Bird's Eye View: Refined Height Layer Points")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.tight_layout()
    # plt.show()


    for h_val in range(1, 6):
        pts = np.array(refined_points_by_height[h_val])
        if len(pts) == 0:
            continue
        xs = pts[:, 0]
        ys = pts[:, 2]  # Z축이 전방

        plt.figure(figsize=(8, 8))
        plt.scatter(xs, ys, s=4, color=cmap(h_val - 1))
        plt.xlabel('X axis [m]')
        plt.ylabel('Z axis [m] (depth forward)')
        plt.title(f"Bird's Eye View - {h_val}m Layer")
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()

        save_path = f"bev_layer_{h_val}m.png"
        # plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved {save_path}")

    from scipy.spatial import KDTree

    # 연결 및 시각화
    all_layers_lines = {}  # 루프 밖으로 이동하여 모든 layer를 저장
    for h_val in range(1, 6):
        pts = np.array(refined_points_by_height[h_val])
        if len(pts) < 2:
            continue

        # Z가 25m 이상인 점 제거
        pts = pts[pts[:, 2] <= 25.0]
        if len(pts) < 2:
            continue

        tree = KDTree(pts[:, [0, 2]])
        connected_indices = set()
        raw_connections = []

        for i, pt in enumerate(pts):
            dist, idx = tree.query(pt[[0, 2]], k=2)
            nearest_idx = idx[1]
            nearest_dist = dist[1]

            z = pt[2]
            if z < 5:
                threshold = 0.5
            elif z < 10:
                threshold = 0.8
            elif z < 15:
                threshold = 1.0
            elif z < 25:
                threshold = 1.5
            else:
                continue

            if nearest_dist <= threshold:
                raw_connections.append((i, nearest_idx))
                connected_indices.update([i, nearest_idx])

        if len(raw_connections) == 0:
            continue

        connected_indices = sorted(list(connected_indices))
        index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(connected_indices)}

        filtered_pts = pts[connected_indices]
        refined_points_by_height[h_val] = filtered_pts

        remapped_connections = []
        for i, j in raw_connections:
            if i in index_map and j in index_map:
                remapped_connections.append((index_map[i], index_map[j]))
        # === 선 정보 저장 ===
        line_segments = []
        for i, j in remapped_connections:
            pt1 = filtered_pts[i]
            pt2 = filtered_pts[j]
            line_segments.append([pt1[[0, 2]].tolist(), pt2[[0, 2]].tolist()])  # X, Z만 저장

        # numpy로 저장
        lines_filename = f"{prefix}_bev_layer_{h_val}m_lines.npy"
        lines_filepath = os.path.join(layer_save_dir, lines_filename)
        np.save(lines_filepath, np.array(line_segments, dtype=np.float32))
        print(f"[INFO] 선 정보 저장 완료: {lines_filepath}")

        # 시각화: 점 없이 선만 그리기
        plt.figure(figsize=(8, 8))
        for i, j in remapped_connections:
            pt1 = filtered_pts[i]
            pt2 = filtered_pts[j]
            plt.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], color='black', linewidth=1.5)

        plt.xlabel('X axis [m]')
        plt.ylabel('Z axis [m] (depth forward)')
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-16, 16)
        plt.ylim(0, 32)
        plt.tight_layout()

        save_path = f"bev_connected_layer_{h_val}m_lines_only.png"
        # plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved {save_path}")

        # === 1. 이미지 이름에서 prefix 추출 ===
        image_basename = os.path.basename(IMAGE_PATH)  # e.g. '08630.jpg'
        prefix = os.path.splitext(image_basename)[0]   # '08630'
        # === 2. 저장 디렉토리 설정 및 생성 ===
        layer_save_dir = os.path.join(SAVE_DIR, 'contours', prefix)
        os.makedirs(layer_save_dir, exist_ok=True)

        # === 3. 각 레이어별로 저장 ===
        for h_val in range(1, 6):
            pts = np.array(refined_points_by_height[h_val])
            if len(pts) == 0:
                continue

            pts = pts[pts[:, 2] <= 25.0]
            if len(pts) == 0:
                continue

            points_2d = pts[:, [0, 2]]  # X-Z 평면

            filename = f"{prefix}_bev_layer_{h_val}m_points.npy"
            filepath = os.path.join(layer_save_dir, filename)
            np.save(filepath, points_2d)

            print(f"[INFO] 저장 완료: {filepath}")

        # 원본 이미지의 파일명 추출 및 복사 대상 경로 설정
        images_dir = os.path.join(SAVE_DIR, "images")
        os.makedirs(images_dir, exist_ok=True)
        image_target_path = os.path.join(images_dir, os.path.basename(IMAGE_PATH))
        shutil.copy(IMAGE_PATH, image_target_path)
        print(f"[INFO] 원본 이미지 복사 완료: {image_target_path}")


if __name__ == "__main__":
    # === 이미지 폴더 설정 ===
    IMAGE_FOLDER = "/home/ohj/GT_4_22/4_19_image"
    DEPTH_FOLDER = "/home/ohj/GT_4_22/4_19_depth"
    MASK_FOLDER = "/home/ohj/"

    # === 이미지 파일 목록 ===
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))])
    if not image_files:
        print("⚠ 이미지가 없습니다.")
        exit()
    while 1:
        # === 사용자에게 이미지 선택 받기 ===
        print("=== 이미지 선택 ===")
        # for idx, fname in enumerate(image_files):
        #     print(f"[{idx}] {fname}")

        selected_idx = int(input("번호 선택: "))
        selected_image = image_files[selected_idx]
        prefix = os.path.splitext(selected_image)[0]

        # === 경로 설정 ===
        IMAGE_PATH = os.path.join(IMAGE_FOLDER, selected_image)
        DEPTH_PATH = os.path.join(DEPTH_FOLDER, f"{prefix}.npy")
        GROUND_MASK_PATH = os.path.join(MASK_FOLDER, f"ground_mask.npy")

        print(f"[INFO] 선택한 이미지: {IMAGE_PATH}")
        print(f"[INFO] 뎁스 경로: {DEPTH_PATH}")
        print(f"[INFO] 마스크 저장 경로: {GROUND_MASK_PATH}")

        # === 실행 ===
        generate_ground_mask(IMAGE_PATH, GROUND_MASK_PATH)
        visualize_3m_plane_points(mode="stereo")
