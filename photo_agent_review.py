import argparse
import json
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from photo_agent_cleanup import delete_staged_images, staged_image_files
from photo_agent_move import move_files


THUMB_SIZE = 220


def load_thumbnail(path: str) -> QPixmap:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        data = img.tobytes("raw", "RGB")
        qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())


class ReviewWindow(QMainWindow):
    def __init__(self, report_path: Path):
        super().__init__()
        self.report_path = report_path
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.groups = self._flatten_groups()
        self.decisions: dict[str, dict] = {}
        self.current_group_key: str | None = None
        self.checkboxes: list[tuple[QCheckBox, dict]] = []
        self.group_items_by_key: dict[str, QListWidgetItem] = {}
        self.setWindowTitle("Photo Agent Review")
        self.resize(1400, 900)
        self._build_ui()
        if self.groups:
            self.group_list.setCurrentRow(0)

    def _flatten_groups(self) -> list[dict]:
        groups = []
        for source_key, label in [("duplicate_groups", "重複"), ("similar_groups", "相似")]:
            for group in self.report.get(source_key, []):
                groups.append(
                    {
                        "key": f"{source_key}:{group['group_id']}",
                        "label": label,
                        "group_id": group["group_id"],
                        "items": group["items"],
                    }
                )
        return groups

    def _build_ui(self) -> None:
        splitter = QSplitter()
        self.group_list = QListWidget()
        for group in self.groups:
            item = QListWidgetItem(f"{group['label']} #{group['group_id']} - {len(group['items'])} 張")
            item.setData(Qt.ItemDataRole.UserRole, group["key"])
            self.group_list.addItem(item)
            self.group_items_by_key[group["key"]] = item
        self.group_list.currentItemChanged.connect(self._show_selected_group)
        splitter.addWidget(self.group_list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.header = QLabel("選擇左側群組開始審閱")
        self.header.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.header.setWordWrap(True)
        right_layout.addWidget(self.header)
        self.status = QLabel("勾選代表保留；未勾選代表待移至暫存。審閱階段不會移動檔案。")
        self.status.setWordWrap(True)
        right_layout.addWidget(self.status)

        actions = QHBoxLayout()
        self.accept_ai_btn = QPushButton("接受 AI 建議並下一組")
        self.keep_one_btn = QPushButton("只保留 1 張並下一組")
        self.apply_manual_btn = QPushButton("套用目前勾選")
        self.skip_btn = QPushButton("跳過並下一組")
        self.save_btn = QPushButton("完成並儲存待移動清單")
        self.delete_staged_btn = QPushButton("刪除暫存相片")
        self.accept_ai_btn.clicked.connect(self._accept_ai)
        self.keep_one_btn.clicked.connect(self._accept_ai)
        self.apply_manual_btn.clicked.connect(self._apply_manual)
        self.skip_btn.clicked.connect(self._skip_group)
        self.save_btn.clicked.connect(self._save_plan)
        self.delete_staged_btn.clicked.connect(self._delete_staged_photos)
        self.delete_staged_btn.setStyleSheet("color: #b00020; font-weight: bold;")
        for button in [self.accept_ai_btn, self.keep_one_btn, self.apply_manual_btn, self.skip_btn, self.save_btn, self.delete_staged_btn]:
            actions.addWidget(button)
        right_layout.addLayout(actions)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.scroll.setWidget(self.grid_container)
        right_layout.addWidget(self.scroll)

        splitter.addWidget(right)
        splitter.setSizes([320, 1080])
        self.setCentralWidget(splitter)

    def _current_group(self) -> dict | None:
        item = self.group_list.currentItem()
        if item is None:
            return None
        key = item.data(Qt.ItemDataRole.UserRole)
        return next((group for group in self.groups if group["key"] == key), None)

    def _clear_items(self) -> None:
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            widget = child.widget()
            if widget:
                widget.deleteLater()
        self.checkboxes = []

    def _show_selected_group(self) -> None:
        group = self._current_group()
        if not group:
            return
        self.current_group_key = group["key"]
        self._clear_items()
        decision = self.decisions.get(group["key"])
        keep_paths = set(decision["keep_paths"]) if decision else {item["path"] for item in group["items"]}
        self.header.setText(f"{group['label']}群組 #{group['group_id']}，共 {len(group['items'])} 張")
        if decision:
            self.status.setText(f"目前狀態：{decision['status']}，待移至暫存 {len(decision['move_paths'])} 張。勾選代表保留。")
        else:
            self.status.setText("尚未套用決策。勾選代表保留；未勾選代表待移至暫存。")

        for index, item in enumerate(group["items"]):
            row = QWidget()
            row_layout = QVBoxLayout(row)
            thumb = QLabel()
            try:
                thumb.setPixmap(load_thumbnail(item["path"]))
            except Exception as exc:
                thumb.setText(f"縮圖載入失敗：{exc}")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setFixedHeight(THUMB_SIZE + 20)
            row_layout.addWidget(thumb)

            keep = QCheckBox("保留")
            keep.setChecked(item["path"] in keep_paths)
            self.checkboxes.append((keep, item))
            row_layout.addWidget(keep)

            tag = "AI 推薦" if index == 0 else ""
            path = Path(item["path"])
            short_path = f"...\\{path.parent.name}\\{path.name}"
            text = QLabel(
                f"{tag}\n{item['name']}\n分數：{item['combined_score']:.3f}\n"
                f"清晰度：{item['sharpness']:.1f}\n曝光：{item['exposure']:.3f}\n{short_path}"
            )
            text.setWordWrap(True)
            text.setToolTip(item["path"])
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row_layout.addWidget(text)
            row.setMinimumWidth(280)
            self.grid_layout.addWidget(row, index // 3, index % 3)

    def _mark_current(self, keep_paths: set[str], status: str) -> None:
        group = self._current_group()
        if not group:
            return
        self.decisions[group["key"]] = {
            "status": status,
            "keep_paths": sorted(keep_paths),
            "move_paths": sorted({item["path"] for item in group["items"]} - keep_paths),
        }
        self._update_group_item_label(group)
        self._show_selected_group()

    def _update_group_item_label(self, group: dict) -> None:
        item = self.group_items_by_key.get(group["key"])
        decision = self.decisions.get(group["key"])
        if not item or not decision:
            return
        prefix = "已審" if decision["status"] == "reviewed" else "跳過"
        item.setText(f"{prefix} - {group['label']} #{group['group_id']} - {len(group['items'])} 張")

    def _go_next_group(self) -> None:
        row = self.group_list.currentRow()
        if row + 1 < self.group_list.count():
            self.group_list.setCurrentRow(row + 1)
        else:
            self.status.setText("已到最後一組。可以按「完成並儲存待移動清單」。")

    def _accept_ai(self) -> None:
        group = self._current_group()
        if group and group["items"]:
            self._mark_current({group["items"][0]["path"]}, "reviewed")
            self._go_next_group()

    def _apply_manual(self) -> None:
        keep_paths = {item["path"] for checkbox, item in self.checkboxes if checkbox.isChecked()}
        self._mark_current(keep_paths, "reviewed")
        self.status.setText(f"已套用目前勾選：保留 {len(keep_paths)} 張。若要繼續，請選下一組或按「完成並儲存」。")

    def _skip_group(self) -> None:
        group = self._current_group()
        if group:
            self.decisions[group["key"]] = {"status": "skipped", "keep_paths": [], "move_paths": []}
            self._update_group_item_label(group)
            self._show_selected_group()
            self._go_next_group()

    def _save_plan(self) -> None:
        reviewed = [decision for decision in self.decisions.values() if decision["status"] == "reviewed"]
        keep_paths = set()
        move_paths = set()
        for decision in reviewed:
            keep_paths.update(decision["keep_paths"])
            move_paths.update(decision["move_paths"])
        move_paths -= keep_paths

        plan_path = self.report_path.parent / "move_plan.json"
        plan = {
            "photo_dir": self.report["photo_dir"],
            "report": str(self.report_path),
            "reviewed_group_count": len(reviewed),
            "skipped_group_count": sum(1 for d in self.decisions.values() if d["status"] == "skipped"),
            "move_count": len(move_paths),
            "move_paths": sorted(move_paths),
            "keep_paths": sorted(keep_paths),
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        answer = QMessageBox.question(
            self,
            "已儲存待移動清單",
            f"已儲存待移動清單：\n{plan_path}\n\n待移動 {len(move_paths)} 個檔案。\n\n是否現在移至暫存資料夾？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            QMessageBox.information(self, "暫不移動", "已保留清單，尚未移動任何檔案。下次按「完成並儲存待移動清單」會再次詢問。")
            return

        try:
            log = move_files(plan_path)
        except Exception as exc:
            QMessageBox.critical(self, "移動失敗", f"移動時發生錯誤：\n{exc}")
            return

        QMessageBox.information(
            self,
            "移動完成",
            f"已移至暫存資料夾：{log['moved_count']} 個檔案\n"
            f"略過：{log['skipped_count']} 個\n"
            f"錯誤：{log['error_count']} 個\n\n"
            f"目標資料夾：\n{log['destination_root']}",
        )

    def _delete_staged_photos(self) -> None:
        staging_dir = Path(self.report["photo_dir"]) / "_photo_agent_staging"
        files = staged_image_files(staging_dir)
        list_path = staging_dir / "pending_delete_list.txt"
        list_path.write_text("\n".join(str(path) for path in files), encoding="utf-8")

        if not files:
            QMessageBox.information(self, "沒有可刪除相片", "暫存資料夾中沒有找到圖片檔。")
            return

        QMessageBox.warning(
            self,
            "永久刪除前確認",
            f"即將列入永久刪除的暫存相片共 {len(files)} 個。\n\n"
            f"完整清單已寫入：\n{list_path}\n\n"
            f"確認永久刪除 {len(files)} 個檔案？此操作無法復原。(yes/no)",
        )
        text, ok = QInputDialog.getText(
            self,
            "輸入 yes 才會刪除",
            f"確認永久刪除 {len(files)} 個檔案？此操作無法復原。(yes/no)",
        )
        if not ok or text != "yes":
            QMessageBox.information(self, "已中止", "未收到精確的 yes，已中止刪除。沒有刪除任何檔案。")
            return

        result = delete_staged_images(staging_dir, text)
        if result.get("errors"):
            QMessageBox.critical(
                self,
                "刪除完成但有錯誤",
                f"已刪除 {result['deleted_count']} 個檔案，錯誤 {len(result['errors'])} 個。\n"
                f"請查看：\n{staging_dir / 'delete_log.json'}",
            )
            return
        QMessageBox.information(
            self,
            "刪除完成",
            f"已永久刪除暫存相片 {result['deleted_count']} 個。\n"
            f"釋放空間：約 {result['deleted_bytes'] / (1024 * 1024):.1f} MB\n"
            f"紀錄：\n{staging_dir / 'delete_log.json'}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    pillow_heif.register_heif_opener()
    app = QApplication([])
    window = ReviewWindow(Path(args.report))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
