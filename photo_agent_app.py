import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pillow_heif
import torch
from PIL import Image, ImageOps
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from photo_agent_cleanup import delete_staged_images, staged_image_files
from photo_agent_move import move_files
from photo_agent_process import IMAGE_EXTENSIONS, iter_images


THUMB_SIZE = 210


def pixmap_for_path(path: str, size: int = THUMB_SIZE) -> QPixmap:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((size, size))
        data = img.tobytes("raw", "RGB")
        qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())


class IndexWorker(QThread):
    line = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, photo_dir: str, model_name: str, batch_size: int, phash_threshold: int, dbscan_eps: float):
        super().__init__()
        self.photo_dir = photo_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.phash_threshold = phash_threshold
        self.dbscan_eps = dbscan_eps

    def run(self) -> None:
        cmd = [
            sys.executable,
            "photo_agent_process.py",
            "--photo-dir",
            self.photo_dir,
            "--batch-size",
            str(self.batch_size),
            "--phash-threshold",
            str(self.phash_threshold),
            "--dbscan-eps",
            str(self.dbscan_eps),
            "--model-name",
            self.model_name,
        ]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).resolve().parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            report_path = ""
            for raw_line in process.stdout or []:
                line = raw_line.strip()
                self.line.emit(line)
                match = re.search(r"processed_new=(\d+)/(\d+)", line)
                if match:
                    self.progress.emit(int(match.group(1)), int(match.group(2)))
                if line.startswith("report="):
                    report_path = line.split("=", 1)[1]
            code = process.wait()
            if code == 0 and report_path:
                self.finished_ok.emit(report_path)
            else:
                self.failed.emit(f"Index process exited with code {code}.")
        except Exception as exc:
            self.failed.emit(repr(exc))


class PhotoAgentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        pillow_heif.register_heif_opener()
        self.setWindowTitle("Photo Agent - DINOv2 Photo Sort")
        self.resize(1500, 950)
        self.photo_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.report_path: Path | None = None
        self.report: dict | None = None
        self.groups: list[dict] = []
        self.decisions: dict[str, dict] = {}
        self.checkboxes: list[tuple[QCheckBox, dict]] = []
        self.group_items_by_key: dict[str, QListWidgetItem] = {}
        self.worker: IndexWorker | None = None
        self._build_ui()
        self._refresh_device_status()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        setup = QGroupBox("1. 選擇資料夾與模型設定")
        setup_layout = QGridLayout(setup)
        self.folder_edit = QLineEdit()
        self.choose_btn = QPushButton("打開要整理的相片資料夾")
        self.choose_btn.clicked.connect(self._choose_folder)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            [
                "facebook/dinov2-large",
                "facebook/dinov2-base",
                "facebook/dinov2-small",
                "facebook/dinov2-giant",
            ]
        )
        self.model_combo.setCurrentText("facebook/dinov2-large")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        self.phash_spin = QSpinBox()
        self.phash_spin.setRange(0, 64)
        self.phash_spin.setValue(10)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.01, 1.0)
        self.eps_spin.setDecimals(3)
        self.eps_spin.setSingleStep(0.01)
        self.eps_spin.setValue(0.15)
        self.device_label = QLabel("")
        self.start_index_btn = QPushButton("2. 開始建立索引")
        self.start_index_btn.clicked.connect(self._start_index)

        setup_layout.addWidget(QLabel("相片資料夾"), 0, 0)
        setup_layout.addWidget(self.folder_edit, 0, 1, 1, 5)
        setup_layout.addWidget(self.choose_btn, 0, 6)
        setup_layout.addWidget(QLabel("DINOv2 模型"), 1, 0)
        setup_layout.addWidget(self.model_combo, 1, 1, 1, 2)
        setup_layout.addWidget(QLabel("Batch"), 1, 3)
        setup_layout.addWidget(self.batch_spin, 1, 4)
        setup_layout.addWidget(QLabel("pHash 閾值"), 1, 5)
        setup_layout.addWidget(self.phash_spin, 1, 6)
        setup_layout.addWidget(QLabel("DBSCAN eps"), 2, 0)
        setup_layout.addWidget(self.eps_spin, 2, 1)
        setup_layout.addWidget(self.device_label, 2, 2, 1, 3)
        setup_layout.addWidget(self.start_index_btn, 2, 6)
        root_layout.addWidget(setup)

        progress_box = QGroupBox("3. 索引流程")
        progress_layout = QHBoxLayout(progress_box)
        self.progress = QProgressBar()
        self.index_thumb = QLabel("索引時會顯示目前處理附近的縮圖")
        self.index_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_thumb.setFixedSize(240, 240)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        progress_layout.addWidget(self.index_thumb)
        progress_right = QVBoxLayout()
        progress_right.addWidget(self.progress)
        progress_right.addWidget(self.log)
        progress_layout.addLayout(progress_right, stretch=1)
        root_layout.addWidget(progress_box)

        review_splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        filter_box = QGroupBox("4. 左側篩選項目")
        filter_layout = QVBoxLayout(filter_box)
        self.dup_filter = QCheckBox("pHash 重複群組")
        self.sim_filter = QCheckBox("DINOv2 語義相似群組")
        self.dup_filter.setChecked(True)
        self.sim_filter.setChecked(True)
        self.dup_filter.stateChanged.connect(self._populate_groups)
        self.sim_filter.stateChanged.connect(self._populate_groups)
        filter_layout.addWidget(self.dup_filter)
        filter_layout.addWidget(self.sim_filter)
        self.ai_recommend_btn = QPushButton("一鍵 AI 推薦")
        self.ai_recommend_btn.clicked.connect(self._ai_recommend_selected_filters)
        filter_layout.addWidget(self.ai_recommend_btn)
        left_layout.addWidget(filter_box)
        self.group_list = QListWidget()
        self.group_list.currentItemChanged.connect(self._show_group)
        left_layout.addWidget(self.group_list, stretch=1)
        review_splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.review_header = QLabel("索引完成後會在這裡開始篩選流程")
        self.review_header.setWordWrap(True)
        self.review_header.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.review_status = QLabel("一鍵 AI 推薦只有在左側至少勾選一個篩選項目時才會生效。")
        self.review_status.setWordWrap(True)
        right_layout.addWidget(self.review_header)
        right_layout.addWidget(self.review_status)
        actions = QHBoxLayout()
        self.accept_ai_btn = QPushButton("接受本組 AI 建議並下一組")
        self.apply_manual_btn = QPushButton("套用目前勾選")
        self.skip_btn = QPushButton("跳過並下一組")
        self.save_btn = QPushButton("完成並儲存清單")
        self.delete_btn = QPushButton("刪除暫存相片")
        self.delete_btn.setStyleSheet("color: #b00020; font-weight: bold;")
        self.accept_ai_btn.clicked.connect(self._accept_current_ai)
        self.apply_manual_btn.clicked.connect(self._apply_manual)
        self.skip_btn.clicked.connect(self._skip_group)
        self.save_btn.clicked.connect(self._save_plan)
        self.delete_btn.clicked.connect(self._delete_staged_photos)
        for button in [self.accept_ai_btn, self.apply_manual_btn, self.skip_btn, self.save_btn, self.delete_btn]:
            actions.addWidget(button)
        right_layout.addLayout(actions)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.cards_widget = QWidget()
        self.cards_layout = QGridLayout(self.cards_widget)
        self.scroll.setWidget(self.cards_widget)
        right_layout.addWidget(self.scroll, stretch=1)
        review_splitter.addWidget(right)
        review_splitter.setSizes([360, 1140])
        root_layout.addWidget(review_splitter, stretch=1)
        self.setCentralWidget(root)

    def _refresh_device_status(self) -> None:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.device_label.setText(f"GPU 可用：{name} / VRAM {gb:.1f} GB")
        else:
            self.device_label.setText("GPU 不可用，將使用 CPU")

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇要整理的相片資料夾")
        if not folder:
            return
        self.photo_dir = Path(folder)
        self.folder_edit.setText(folder)
        staging = self.photo_dir / "_photo_agent_staging"
        staging.mkdir(exist_ok=True)
        (staging / "duplicates").mkdir(exist_ok=True)
        (staging / "similar_groups").mkdir(exist_ok=True)
        self.image_paths = iter_images(self.photo_dir, staging)
        self.log.append(f"已選擇資料夾：{folder}")
        self.log.append(f"找到圖片：{len(self.image_paths)} 張")

    def _start_index(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "需要資料夾", "請先打開要整理的相片資料夾。")
            return
        self.photo_dir = Path(folder)
        staging = self.photo_dir / "_photo_agent_staging"
        staging.mkdir(exist_ok=True)
        (staging / "duplicates").mkdir(exist_ok=True)
        (staging / "similar_groups").mkdir(exist_ok=True)
        self.image_paths = iter_images(self.photo_dir, staging)
        self.progress.setValue(0)
        self.log.append("開始建立索引...")
        self.start_index_btn.setEnabled(False)
        self.worker = IndexWorker(
            folder,
            self.model_combo.currentText().strip() or "facebook/dinov2-large",
            self.batch_spin.value(),
            self.phash_spin.value(),
            self.eps_spin.value(),
        )
        self.worker.line.connect(self._append_log)
        self.worker.progress.connect(self._update_index_progress)
        self.worker.finished_ok.connect(self._index_finished)
        self.worker.failed.connect(self._index_failed)
        self.worker.start()

    def _append_log(self, line: str) -> None:
        self.log.append(line)

    def _update_index_progress(self, current: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(current)
        if self.image_paths:
            path = self.image_paths[min(current, len(self.image_paths)) - 1]
            try:
                self.index_thumb.setPixmap(pixmap_for_path(str(path), 220))
            except Exception:
                self.index_thumb.setText(path.name)

    def _index_finished(self, report_path: str) -> None:
        self.start_index_btn.setEnabled(True)
        self.report_path = Path(report_path)
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.review_header.setText(
            f"索引完成：{self.report['image_count']} 張，"
            f"重複群組 {self.report['duplicate_group_count']}，"
            f"相似群組 {self.report['similar_group_count']}"
        )
        self._load_groups_from_report()
        self._populate_groups()
        QMessageBox.information(self, "索引完成", "已完成索引，現在可以開始篩選流程。")

    def _index_failed(self, message: str) -> None:
        self.start_index_btn.setEnabled(True)
        QMessageBox.critical(self, "索引失敗", message)

    def _load_groups_from_report(self) -> None:
        self.groups = []
        if not self.report:
            return
        for source_key, label in [("duplicate_groups", "pHash 重複"), ("similar_groups", "DINOv2 相似")]:
            for group in self.report.get(source_key, []):
                self.groups.append(
                    {
                        "key": f"{source_key}:{group['group_id']}",
                        "type": source_key,
                        "label": label,
                        "group_id": group["group_id"],
                        "items": group["items"],
                    }
                )

    def _selected_filter_types(self) -> set[str]:
        selected = set()
        if self.dup_filter.isChecked():
            selected.add("duplicate_groups")
        if self.sim_filter.isChecked():
            selected.add("similar_groups")
        return selected

    def _populate_groups(self) -> None:
        self.group_list.clear()
        self.group_items_by_key = {}
        selected = self._selected_filter_types()
        for group in self.groups:
            if group["type"] not in selected:
                continue
            decision = self.decisions.get(group["key"])
            prefix = ""
            if decision:
                prefix = "已審 - " if decision["status"] == "reviewed" else "跳過 - "
            item = QListWidgetItem(f"{prefix}{group['label']} #{group['group_id']} - {len(group['items'])} 張")
            item.setData(Qt.ItemDataRole.UserRole, group["key"])
            self.group_list.addItem(item)
            self.group_items_by_key[group["key"]] = item
        if self.group_list.count():
            self.group_list.setCurrentRow(0)
        else:
            self._clear_cards()

    def _current_group(self) -> dict | None:
        item = self.group_list.currentItem()
        if not item:
            return None
        key = item.data(Qt.ItemDataRole.UserRole)
        return next((group for group in self.groups if group["key"] == key), None)

    def _clear_cards(self) -> None:
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            widget = child.widget()
            if widget:
                widget.deleteLater()
        self.checkboxes = []

    def _show_group(self) -> None:
        group = self._current_group()
        self._clear_cards()
        if not group:
            return
        decision = self.decisions.get(group["key"])
        keep_paths = set(decision["keep_paths"]) if decision else {item["path"] for item in group["items"]}
        self.review_header.setText(f"{group['label']}群組 #{group['group_id']}，共 {len(group['items'])} 張")
        self.review_status.setText("勾選代表保留，未勾選代表待移至暫存。")
        for index, item in enumerate(group["items"]):
            card = QWidget()
            layout = QVBoxLayout(card)
            thumb = QLabel()
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setFixedHeight(THUMB_SIZE + 20)
            try:
                thumb.setPixmap(pixmap_for_path(item["path"]))
            except Exception as exc:
                thumb.setText(f"縮圖載入失敗：{exc}")
            layout.addWidget(thumb)
            keep = QCheckBox("保留")
            keep.setChecked(item["path"] in keep_paths)
            layout.addWidget(keep)
            self.checkboxes.append((keep, item))
            tag = "AI 推薦\n" if index == 0 else ""
            path = Path(item["path"])
            text = QLabel(
                f"{tag}{item['name']}\n"
                f"分數：{item['combined_score']:.3f}\n"
                f"清晰度：{item['sharpness']:.1f} / 曝光：{item['exposure']:.3f}\n"
                f"...\\{path.parent.name}\\{path.name}"
            )
            text.setWordWrap(True)
            text.setToolTip(item["path"])
            layout.addWidget(text)
            card.setMinimumWidth(280)
            self.cards_layout.addWidget(card, index // 3, index % 3)

    def _mark_group(self, group: dict, keep_paths: set[str], status: str = "reviewed") -> None:
        self.decisions[group["key"]] = {
            "status": status,
            "keep_paths": sorted(keep_paths),
            "move_paths": sorted({item["path"] for item in group["items"]} - keep_paths),
        }
        self._populate_groups()

    def _go_next_group(self) -> None:
        row = self.group_list.currentRow()
        if row + 1 < self.group_list.count():
            self.group_list.setCurrentRow(row + 1)

    def _accept_current_ai(self) -> None:
        group = self._current_group()
        if not group or not group["items"]:
            return
        self.decisions[group["key"]] = {
            "status": "reviewed",
            "keep_paths": [group["items"][0]["path"]],
            "move_paths": sorted(item["path"] for item in group["items"][1:]),
        }
        self._populate_groups()
        self._go_next_group()

    def _apply_manual(self) -> None:
        group = self._current_group()
        if not group:
            return
        keep_paths = {item["path"] for checkbox, item in self.checkboxes if checkbox.isChecked()}
        self.decisions[group["key"]] = {
            "status": "reviewed",
            "keep_paths": sorted(keep_paths),
            "move_paths": sorted({item["path"] for item in group["items"]} - keep_paths),
        }
        self._populate_groups()

    def _skip_group(self) -> None:
        group = self._current_group()
        if not group:
            return
        self.decisions[group["key"]] = {"status": "skipped", "keep_paths": [], "move_paths": []}
        self._populate_groups()
        self._go_next_group()

    def _ai_recommend_selected_filters(self) -> None:
        selected = self._selected_filter_types()
        if not selected:
            self.review_status.setText("未勾選左側篩選項目，一鍵 AI 推薦不會執行。")
            return
        count = 0
        for group in self.groups:
            if group["type"] not in selected or not group["items"]:
                continue
            self.decisions[group["key"]] = {
                "status": "reviewed",
                "keep_paths": [group["items"][0]["path"]],
                "move_paths": sorted(item["path"] for item in group["items"][1:]),
            }
            count += 1
        self._populate_groups()
        self.review_status.setText(f"一鍵 AI 推薦已套用 {count} 個已勾選篩選項目的群組。")

    def _save_plan(self) -> None:
        if not self.report or not self.report_path:
            QMessageBox.warning(self, "尚未索引", "請先完成索引。")
            return
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
            "已儲存清單",
            f"已儲存待移動清單：\n{plan_path}\n\n待移動 {len(move_paths)} 個檔案。\n\n是否現在移至暫存資料夾？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.review_status.setText("已儲存清單，尚未移動。下次按完成並儲存清單會再次詢問。")
            return
        log = move_files(plan_path)
        QMessageBox.information(
            self,
            "移動完成",
            f"已移至暫存資料夾：{log['moved_count']} 個\n錯誤：{log['error_count']} 個\n{log['destination_root']}",
        )

    def _delete_staged_photos(self) -> None:
        if not self.photo_dir and self.folder_edit.text().strip():
            self.photo_dir = Path(self.folder_edit.text().strip())
        if not self.photo_dir:
            QMessageBox.warning(self, "需要資料夾", "請先選擇相片資料夾。")
            return
        staging_dir = self.photo_dir / "_photo_agent_staging"
        files = staged_image_files(staging_dir)
        staging_dir.mkdir(exist_ok=True)
        list_path = staging_dir / "pending_delete_list.txt"
        list_path.write_text("\n".join(str(path) for path in files), encoding="utf-8")
        if not files:
            QMessageBox.information(self, "沒有可刪除相片", "暫存資料夾中沒有圖片檔。")
            return
        QMessageBox.warning(
            self,
            "永久刪除前確認",
            f"確認永久刪除暫存區中的 {len(files)} 個檔案？此操作無法復原。(yes/no)\n\n完整清單：\n{list_path}",
        )
        text, ok = QInputDialog.getText(
            self,
            "輸入 yes 才會刪除",
            f"確認永久刪除暫存區中的 {len(files)} 個檔案？此操作無法復原。(yes/no)",
        )
        if not ok or text != "yes":
            QMessageBox.information(self, "已中止", "未收到精確 yes，已中止刪除。")
            return
        result = delete_staged_images(staging_dir, text)
        QMessageBox.information(
            self,
            "刪除完成",
            f"刪除 {result['deleted_count']} 個檔案，釋放約 {result['deleted_bytes'] / (1024 * 1024):.1f} MB。",
        )


def main() -> int:
    app = QApplication([])
    window = PhotoAgentApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
